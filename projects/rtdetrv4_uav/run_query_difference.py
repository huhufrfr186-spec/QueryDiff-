"""Finite ten-class VisDrone pilot. Separate from the seven-class legacy runs."""
from bootstrap import ROOT, write_json, sha256
import argparse, json, random, sys, time, traceback, hashlib
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from PIL import Image, ImageEnhance
from torchvision.transforms.functional import pil_to_tensor
from transformers import CLIPModel
import train_visdrone as base
from run_feature_upsampling import TEACHER, UPSTREAM
from query_difference import object_targets, graph_edges, losses, gam_ratio, gam_update


class TenClassData:
    def __init__(self, split, limit, seed):
        self.path = base.DATA/f'{split}.json'
        # Only image membership is inherited, all ten labels come from original GT.
        self.rows = sorted(json.loads(self.path.read_text())['images'], key=lambda x:x['file_name'])
        random.Random(seed).shuffle(self.rows)
        if limit: self.rows = self.rows[:limit]
        self.raw = {}
        for row in self.rows:
            path = ROOT/'data/VisDrone2019-DET-train/annotations'/Path(row['file_name']).with_suffix('.txt')
            self.raw[row['id']] = np.array([[float(v) for v in line.split(',')[:8]]
                for line in path.read_text().splitlines() if line.strip()], dtype=np.float32).reshape(-1,8)

    def sample(self, row, device, augment=False, condition='clean'):
        with Image.open(base.IMAGES/row['file_name']) as handle: image=handle.convert('RGB')
        flip = augment and random.random()<.5
        if flip: image=image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if augment:
            image=ImageEnhance.Brightness(image).enhance(random.uniform(.8,1.2))
            image=ImageEnhance.Contrast(image).enhance(random.uniform(.8,1.2))
        image=image.resize((640,640),Image.Resampling.BILINEAR)
        if condition.startswith('down'):
            size=320 if condition=='down2' else 160
            image=image.resize((size,size),Image.Resampling.BILINEAR).resize((640,640),Image.Resampling.BILINEAR)
        elif condition.startswith('blur'):
            from PIL import ImageFilter
            image=image.filter(ImageFilter.GaussianBlur(1 if condition=='blur1' else 2))
        boxes=[]; labels=[]
        for x,y,w,h,score,category,_,_ in self.raw[row['id']]:
            if score<=0 or not 1<=category<=10: continue
            x2,y2=min(row['width'],x+w),min(row['height'],y+h)
            x,y=max(0,x),max(0,y)
            if x2<=x or y2<=y: continue
            cx,cy=(x+x2)/(2*row['width']),(y+y2)/(2*row['height'])
            boxes.append([1-cx if flip else cx,cy,(x2-x)/row['width'],(y2-y)/row['height']])
            labels.append(int(category)-1)
        x=pil_to_tensor(image).float().div(255).unsqueeze(0).to(device)
        return x,[dict(boxes=torch.tensor(boxes,dtype=torch.float32,device=device).reshape(-1,4),
                       labels=torch.tensor(labels,dtype=torch.long,device=device))]


class Teacher:
    def __init__(self, device):
        container=CLIPModel.from_pretrained(str(TEACHER),local_files_only=True)
        self.vision=container.vision_model.to(device).eval().requires_grad_(False)
        if device=='cuda': self.vision.half()
        pre=json.loads((TEACHER/'preprocessor_config.json').read_text())
        self.mean=torch.tensor(pre['image_mean'],device=device)[None,:,None,None]
        self.std=torch.tensor(pre['image_std'],device=device)[None,:,None,None]
        sys.path.insert(0,str(UPSTREAM))
        from anyup.model import AnyUp
        self.up=AnyUp(use_natten=False)
        self.up.load_state_dict(torch.load(UPSTREAM/'anyup_paper.pth',map_location='cpu',weights_only=True))
        self.up=self.up.to(device).eval().requires_grad_(False)
        self.gmean=torch.tensor([.485,.456,.406],device=device)[None,:,None,None]
        self.gstd=torch.tensor([.229,.224,.225],device=device)[None,:,None,None]
        self.cache=ROOT/'results/querydiff_teacher_cache_20261012'
        self.cache.mkdir(exist_ok=True)
        self.signature=sha256(__file__)+sha256(TEACHER/'pytorch_model.bin')+sha256(UPSTREAM/'anyup_paper.pth')
        self.cache_hits=self.cache_misses=0

    @torch.no_grad()
    def __call__(self, x, boxes):
        key=hashlib.sha256(self.signature.encode()+x.cpu().numpy().tobytes()+boxes.cpu().numpy().tobytes()).hexdigest()
        path=self.cache/(key+'.pth')
        if path.exists():
            saved=torch.load(path,map_location=x.device,weights_only=True)
            self.cache_hits+=1
            return saved['coarse'].float(),saved['objects'][0],saved['objects'][1:]
        objects=[]; coarse=None
        # Identical clean + two geometry-preserving views for every arm.
        for scale in (1.,.95,1.05):
            image=(x*scale).clamp(0,1)
            tokens=self.vision(pixel_values=((image-self.mean)/self.std).to(next(self.vision.parameters()).dtype),
                               interpolate_pos_encoding=True).last_hidden_state[:,1:]
            feature=tokens.float().transpose(1,2).reshape(1,768,40,40)
            fine=self.up((image-self.gmean)/self.gstd,feature,output_size=(80,80),q_chunk_size=64)
            objects.append(object_targets(fine,boxes))
            if scale==1.: coarse=feature
        saved=dict(coarse=coarse.half().cpu(),objects=[v.cpu() for v in objects])
        temporary=path.with_suffix('.tmp');torch.save(saved,temporary);temporary.replace(path)
        self.cache_misses+=1
        return coarse,objects[0],objects[1:]


def build(device):
    model,criterion,post=base.build_model()
    # Same seed gives the same random ten-class heads in every arm.
    model.decoder.enc_score_head=nn.Linear(model.decoder.enc_score_head.in_features,10)
    model.decoder.dec_score_head=nn.ModuleList([nn.Linear(h.in_features,10) for h in model.decoder.dec_score_head])
    for head in [model.decoder.enc_score_head,*model.decoder.dec_score_head]:
        nn.init.constant_(head.bias,-4.59511985)
    model.decoder.denoising_class_embed=nn.Embedding(11,model.decoder.denoising_class_embed.embedding_dim,padding_idx=10)
    model.decoder.num_classes=criterion.num_classes=post.num_classes=10
    return model.to(device),criterion.to(device),post.to(device)


@torch.inference_mode()
def evaluate(model,post,data,out,device,epoch,condition):
    model.eval(); detections=[]; gt=[]; hw=[]
    tool=next((ROOT/'work/visdrone-det-toolkit-python-api').glob('*/viseval'))
    sys.path.insert(0,str(tool.parent)); np.int=int
    from viseval.eval_det import eval_det
    for row in data.rows:
        x,_=data.sample(row,device,condition=condition)
        pred=post(model(x),torch.tensor([[row['width'],row['height']]],device=device))[0]
        box=pred['boxes'].cpu().numpy(); box[:,2:]-=box[:,:2]
        det=np.column_stack([box,pred['scores'].cpu().numpy(),pred['labels'].cpu().numpy()+1])
        det=det[np.argsort(-det[:,4])]
        detections.append(det);gt.append(data.raw[row['id']].astype(np.int32));hw.append((row['height'],row['width']))
    result=eval_det(gt,detections,[v[0] for v in hw],[v[1] for v in hw],per_class=True)
    keys=['AP','AP50','AP75','AR1','AR10','AR100','AR500']
    metrics={key:float(result[i]) for i,key in enumerate(keys)}
    metrics.update(images=len(data.rows),epoch=epoch,condition=condition,
        protocol='Ten-class internal development subset, VisDrone toolkit Python port, top300 outputs; not official test-dev')
    write_json(out/f'metrics_epoch_{epoch:03d}_{condition}.json',metrics)
    write_json(out/f'predictions_epoch_{epoch:03d}_{condition}.json',[
        dict(file_name=row['file_name'],detections=det.tolist()) for row,det in zip(data.rows,detections)])
    return metrics


def main(args):
    out=ROOT/'results'/args.name;out.mkdir(exist_ok=False)
    status={'status':'initializing','mode':args.mode,'epoch':0,'step':0}
    write_json(out/'status.json',status);started=time.time()
    try:
        torch.set_num_threads(2);random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
        torch.backends.cudnn.benchmark=False
        fit=TenClassData('fit',args.fit_limit,args.seed);dev=TenClassData('development',args.dev_limit,args.seed)
        assert not {r['file_name'] for r in fit.rows}&{r['file_name'] for r in dev.rows}
        if args.smoke:
            for row in fit.rows:
                _,sample_targets=fit.sample(row,'cpu')
                if len(graph_edges(sample_targets[0]['boxes'])):
                    fit.rows=[row];break
            else:raise RuntimeError('Smoke selection contains no shared-grid edge; use a larger fit limit')
        model,criterion,post=build(args.device)
        query=nn.Linear(model.decoder.dec_score_head[-1].in_features,768).to(args.device)
        deep=nn.Conv2d(256,768,1).to(args.device)
        captured={}
        def capture_encoder(module,inputs,output):
            if model.training:
                side=int(output.shape[1]**.5)
                captured['deep']=output.transpose(1,2).reshape(output.shape[0],output.shape[2],side,side)
        def capture_query(module,inputs):
            if model.training:captured['query']=inputs[0][:,-model.decoder.num_queries:]
        hooks=[model.encoder.encoder[0].register_forward_hook(capture_encoder),
               model.decoder.dec_score_head[-1].register_forward_pre_hook(capture_query)]
        parameters=list(model.parameters())+list(query.parameters())+list(deep.parameters())
        optimizer=torch.optim.AdamW(parameters,lr=1e-4,weight_decay=1e-4)
        teacher=Teacher(args.device);deep_weight=5.
        manifest=dict(arguments=vars(args),fit=[r['file_name'] for r in fit.rows],development=[r['file_name'] for r in dev.rows],
            source_sha256=sha256(__file__),operator_sha256=sha256(Path(__file__).with_name('query_difference.py')),
            teacher_sha256=sha256(TEACHER/'pytorch_model.bin'),anyup_sha256=sha256(UPSTREAM/'anyup_paper.pth'),
            checkpoint_sha256=sha256(base.WEIGHTS),protocol='ten_class_pilot_not_full_converged_paper_result',
            initialization='Official COCO weights; common randomly initialized ten-class heads; no legacy seven-class checkpoint',
            GAM='Official released odds-ratio update, rho11 delta1; non-DDP module name corrected; extra projection excluded from detector-module denominator',
            query='Ordinary final classification-head input; excludes denoising prefix; final one-to-one matching',
            augmentation='Same flip brightness contrast; fixed .95/1.05 teacher brightness views, all arms',
            cache='Content-addressed exact input pixels, boxes, teacher weights and runner source; same teacher outputs in all arms',
            selection='Final epoch, no development-based checkpoint choice')
        write_json(out/'provenance.json',manifest)
        # Fixed coefficients from the same training-only calibration images.
        # This prevents the inverse-energy units from dominating detection by accident.
        state=(random.getstate(),np.random.get_state(),torch.get_rng_state(),
               torch.cuda.get_rng_state_all() if args.device=='cuda' else None)
        model.train()
        for module in model.modules():
            if isinstance(module,nn.modules.batchnorm._BatchNorm):module.eval()
        calibration=[]
        if not args.smoke:
            with torch.no_grad():
                for row in fit.rows:
                    x,targets=fit.sample(row,args.device)
                    if len(graph_edges(targets[0]['boxes']))==0:continue
                    coarse,t,perturbed=teacher(x,targets[0]['boxes'])
                    outputs=model(x,targets);det_losses=criterion(outputs,targets)
                    qi,ti=[v.to(args.device) for v in criterion.matcher({k:outputs[k] for k in ('pred_logits','pred_boxes')},targets)['indices'][0]]
                    z=query(captured.pop('query')[0,qi]);captured.pop('deep')
                    edge=graph_edges(targets[0]['boxes'][ti])
                    anchor,difference,_=losses(z,t[ti],[v[ti] for v in perturbed],edge,args.mode if args.mode!='B' else 'M')
                    calibration.append([float(sum(det_losses.values())),float(anchor),float(difference)])
                    if len(calibration)==4:break
            random.setstate(state[0]);np.random.set_state(state[1]);torch.set_rng_state(state[2])
            if state[3] is not None:torch.cuda.set_rng_state_all(state[3])
            if not calibration:raise RuntimeError('No shared-grid objects in training pilot')
            det_mean,anchor_mean,diff_mean=np.mean(calibration,axis=0)
            lambda_anchor=args.anchor_weight*.1*det_mean/max(anchor_mean,1e-8)
            lambda_difference=args.diff_weight*.1*det_mean/max(diff_mean,1e-8) if args.mode in ('M','A2') else 0.
            write_json(out/'loss_scale_calibration.json',dict(source='First four shared-grid training images; no development labels or optimizer updates',
                measurements=calibration,lambda_anchor=lambda_anchor,lambda_difference=lambda_difference,
                purpose='Equal initial loss magnitude budget for Q/M/A2; fixed for the entire run'))
        else:lambda_anchor=args.anchor_weight;lambda_difference=args.diff_weight
        for epoch in range(1,args.epochs+1):
            model.train();query.train();deep.train()
            for module in model.modules():
                if isinstance(module,nn.modules.batchnorm._BatchNorm):module.eval()
            rows=list(fit.rows);random.shuffle(rows);optimizer.zero_grad(set_to_none=True);ratios=[];means=[]
            for step,row in enumerate(rows,1):
                if (out/'STOP').exists():raise InterruptedError('Run-local stop requested')
                x,targets=fit.sample(row,args.device,augment=True)
                coarse,t,perturbed=teacher(x,targets[0]['boxes'])
                outputs=model(x,targets);det_losses=criterion(outputs,targets)
                indices=criterion.matcher({k:outputs[k] for k in ('pred_logits','pred_boxes')},targets)['indices'][0]
                qi,ti=[index.to(args.device) for index in indices]
                z=query(captured.pop('query')[0,qi])
                boxes=targets[0]['boxes'][ti]
                edges=graph_edges(boxes)
                anchor,difference,stats=losses(z,t[ti],[v[ti] for v in perturbed],edges,args.mode)
                feature=captured.pop('deep');projected=deep(feature)
                deep_loss=(1-F.cosine_similarity(projected,F.interpolate(coarse,feature.shape[-2:],mode='bilinear',align_corners=False),dim=1)).mean()
                loss=sum(det_losses.values())+deep_weight*deep_loss
                if args.mode!='B':loss=loss+lambda_anchor*anchor+lambda_difference*difference
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite total loss')
                group_size=min(4,len(rows)-((step-1)//4)*4)
                (loss/group_size).backward()
                if step%4==0 or step==len(rows):
                    ratios.append(gam_ratio(model))
                    norm=torch.nn.utils.clip_grad_norm_(parameters,.1)
                    if not torch.isfinite(norm):raise RuntimeError('Nonfinite gradient')
                    optimizer.step();optimizer.zero_grad(set_to_none=True)
                means.append([float(loss.detach()),float(anchor.detach()),float(difference.detach())])
                status.update(status='training',epoch=epoch,step=step,total_steps=len(rows),
                    loss=means[-1][0],anchor=means[-1][1],difference=means[-1][2],**stats,
                    elapsed_seconds=time.time()-started,
                    teacher_cache_hits=teacher.cache_hits,teacher_cache_misses=teacher.cache_misses,
                    peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20 if args.device=='cuda' else None)
                if step%5==0 or step==1 or step==len(rows):write_json(out/'status.json',status);print(json.dumps(status),flush=True)
                del x,targets,coarse,t,perturbed,outputs,det_losses,z,anchor,difference,feature,projected,deep_loss,loss
                if args.smoke:
                    status.update(status='smoke_passed',purpose='Real-image forward/backward integration only, no detection performance claim')
                    write_json(out/'status.json',status);return
            deep_weight=gam_update(deep_weight,float(np.mean(ratios)))
            torch.save(dict(model=model.state_dict(),query=query.state_dict(),deep=deep.state_dict(),optimizer=optimizer.state_dict(),
                            epoch=epoch,provenance=manifest,deep_weight=deep_weight),out/'last.pth')
            write_json(out/'status.json',dict(status,status='evaluating'))
            metrics=evaluate(model,post,dev,out,args.device,epoch,'clean')
            with (out/'history.jsonl').open('a') as handle:handle.write(json.dumps(dict(metrics,mean_losses=np.mean(means,axis=0).tolist(),deep_weight=deep_weight))+'\n')
        conditions={'clean':metrics}
        if args.conditions:
            for condition in ('down2','down4','blur1','blur2'):
                conditions[condition]=evaluate(model,post,dev,out,args.device,args.epochs,condition)
        status.update(status='completed',metrics=metrics,conditions=conditions,elapsed_seconds=time.time()-started)
        write_json(out/'status.json',status)
    except Exception as error:
        status.update(status='failed',error=str(error),traceback=traceback.format_exc());write_json(out/'status.json',status);raise


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--mode',choices=['B','Q','M','A2'],required=True);p.add_argument('--name',required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda');p.add_argument('--smoke',action='store_true')
    p.add_argument('--epochs',type=int,default=3);p.add_argument('--fit-limit',type=int,default=128)
    p.add_argument('--dev-limit',type=int,default=32);p.add_argument('--seed',type=int,default=20261011)
    p.add_argument('--anchor-weight',type=float,default=1.);p.add_argument('--diff-weight',type=float,default=1.)
    p.add_argument('--conditions',action='store_true');a=p.parse_args()
    if Path(a.name).name!=a.name or a.epochs<1 or min(a.fit_limit,a.dev_limit)<0:raise ValueError('Invalid run parameters')
    main(a)
