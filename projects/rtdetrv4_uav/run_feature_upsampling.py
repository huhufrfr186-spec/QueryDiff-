"""Equal-budget P3 distillation transfer of public AnyUp; not an original module."""
from bootstrap import ROOT, write_json, sha256
import argparse, json, random, sys, time, traceback
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from transformers import CLIPModel
import train_visdrone as base

UPSTREAM=ROOT/'research/2026_method_survey_20261010/upstream/anyup_archive'
TEACHER=ROOT/'models/openai--clip-vit-base-patch16/57c216476eefef5ab752ec549e440a49ae4ae5f3'
CHECKPOINT=ROOT/'results/visdrone_rtv4_full_control_recovery_20261008/last.pth'

class TargetMaker:
    def __init__(self, mode):
        self.mode=mode
        container=CLIPModel.from_pretrained(str(TEACHER),local_files_only=True)
        self.teacher=container.vision_model.cuda().half().eval().requires_grad_(False)
        pre=json.loads((TEACHER/'preprocessor_config.json').read_text())
        self.mean=torch.tensor(pre['image_mean'],device='cuda')[None,:,None,None]
        self.std=torch.tensor(pre['image_std'],device='cuda')[None,:,None,None]
        self.upsampler=None
        if mode=='anyup':
            sys.path.insert(0,str(UPSTREAM))
            from anyup.model import AnyUp
            self.upsampler=AnyUp(use_natten=False)
            self.upsampler.load_state_dict(torch.load(UPSTREAM/'anyup_paper.pth',map_location='cpu',weights_only=True),strict=True)
            self.upsampler=self.upsampler.cuda().eval().requires_grad_(False)
        self.guide_mean=torch.tensor([.485,.456,.406],device='cuda')[None,:,None,None]
        self.guide_std=torch.tensor([.229,.224,.225],device='cuda')[None,:,None,None]

    @torch.no_grad()
    def __call__(self,x):
        if self.mode=='none': return None
        # Online teacher sees the exact augmented detector image. No stale cache.
        tokens=self.teacher(pixel_values=((x-self.mean)/self.std).half(),interpolate_pos_encoding=True).last_hidden_state[:,1:]
        features=tokens.float().transpose(1,2).reshape(1,768,40,40)
        if self.mode=='bilinear':
            target=F.interpolate(features,(80,80),mode='bilinear',align_corners=False)
        else:
            target=self.upsampler((x-self.guide_mean)/self.guide_std,features,output_size=(80,80),q_chunk_size=64)
        if not torch.isfinite(target).all(): raise RuntimeError('Nonfinite teacher target')
        return target.detach()

def rng_restore(state):
    random.setstate(state['python']); np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch']); torch.cuda.set_rng_state_all(state['cuda'])

def extended_validate(model,post,dev,out,epoch):
    metrics=base.validate(model,post,dev,out,epoch)
    from faster_coco_eval import COCO, COCOeval_faster
    coco=COCO(); coco.dataset=dict(images=dev.rows,categories=dev.raw['categories'],annotations=dev.raw['annotations'],info={}); coco.createIndex()
    detections=json.loads((out/f'predictions_epoch_{epoch:03d}.json').read_text())
    ev=COCOeval_faster(coco,coco.loadRes(detections),iouType='bbox')
    ev.params.imgIds=[r['id'] for r in dev.rows]; ev.params.catIds=list(range(7))
    ev.evaluate(); ev.accumulate(); ev.summarize()
    metrics.update(APsmall=float(ev.stats[3]),APmedium=float(ev.stats[4]),APlarge=float(ev.stats[5]),ARsmall=float(ev.stats[9]))
    precision=ev.eval['precision']
    metrics['class_AP']={}
    for index in range(7):
        values=precision[:,:,index,0,-1]; valid=values[values>-1]
        metrics['class_AP'][str(index)]=float(valid.mean()) if valid.size else None
    write_json(out/f'validation_epoch_{epoch:03d}.json',metrics)
    return metrics

def run(mode,out,epochs,preflight=False):
    started=time.time(); status=dict(status='initializing',mode=mode,epoch=12,step=0)
    out.mkdir(parents=True,exist_ok=False); write_json(out/'status.json',status)
    hook=None
    try:
        torch.set_num_threads(2); torch.backends.cudnn.benchmark=False
        fit,dev=base.Dataset('fit'),base.Dataset('development')
        previous=torch.load(CHECKPOINT,map_location='cpu',weights_only=False)
        p=previous['provenance']
        if p['fit_sha256']!=sha256(fit.path) or p['known_original_ids']!=base.KNOWN or p['effective_batch']!=4: raise ValueError('Common checkpoint protocol differs')
        model,criterion,post=base.build_model()
        model.load_state_dict(previous['model'],strict=True)
        model,criterion,post=model.cuda(),criterion.cuda(),post.cuda()
        optimizer=torch.optim.AdamW([dict(params=[v for n,v in model.named_parameters() if n.startswith('backbone.')],lr=2e-5),dict(params=[v for n,v in model.named_parameters() if not n.startswith('backbone.')],lr=1e-4)],weight_decay=.0001)
        optimizer.load_state_dict(previous['optimizer'])
        torch.manual_seed(20261010)
        projector=nn.Conv2d(256,768,1).cuda()
        optimizer.add_param_group(dict(params=list(projector.parameters()),lr=1e-4,weight_decay=.0001))
        maker=TargetMaker(mode)
        captured={}
        def capture(module,inputs,output):
            if model.training and mode!='none': captured['p3']=output[0]
        hook=model.encoder.register_forward_hook(capture)
        start_epoch=previous['epoch']; rng_restore(previous['rng']); del previous
        manifest=dict(mode=mode,start_epoch=start_epoch,additional_epochs=epochs,fit_images=len(fit.rows),development_images=len(dev.rows),checkpoint_sha256=sha256(CHECKPOINT),fit_sha256=sha256(fit.path),development_sha256=sha256(dev.path),teacher_sha256=sha256(TEACHER/'pytorch_model.bin'),anyup_weights_sha256=sha256(UPSTREAM/'anyup_paper.pth'),script_sha256=sha256(__file__),coefficient=1.,student='All detector parameters jointly trained; BN running stats frozen',projector='Shared Conv2d 256->768 initialization seed20261010; training-only; trained in KD arms',teacher='Local CLIP-B16; exact online augmented 640 input; FP16 frozen teacher; last_hidden_state patch tokens',anyup='Official original public weights; use_natten=False; FP32; query chunks64; guidance ImageNet normalization',comparison='Identical seeds/checkpoint/optimizer/dataset/augmentation/P3 cosine objective; no-KD arm is additional-training control',selection='Final epoch; AP >= no-KD, AP >= bilinear +0.005, APsmall >= bilinear +0.01',limitations=['Transfer experiment, not exact original RT-DETRv4 DINOv3 recipe','Known-only seven-class COCO-style development; no official VisDrone ignore handling','Not open-world unknown detection or untouched confirmatory testing','AnyUp mechanism belongs to its authors; detector gains unverified'])
        write_json(out/'provenance.json',manifest)
        (out/'runner_executed.py').write_bytes(Path(__file__).read_bytes())
        for epoch in range(start_epoch+1,start_epoch+epochs+1):
            model.train(); projector.train()
            for module in model.modules():
                if isinstance(module,nn.modules.batchnorm._BatchNorm): module.eval()
            rows=list(fit.rows); random.shuffle(rows); optimizer.zero_grad(set_to_none=True)
            total=kd_total=0.
            for i,row in enumerate(rows):
                if (out/'STOP').exists(): raise InterruptedError('Run-local STOP requested')
                x,targets=fit.sample(row,augment=True)
                # Produce target before retaining the student graph to fit 4GB.
                target=maker(x)
                outputs=model(x,targets); losses=criterion(outputs,targets)
                kd=torch.zeros((),device='cuda')
                if mode!='none':
                    feature=captured.pop('p3')
                    if feature.shape[-2:]!=(80,80): raise ValueError('Unexpected P3 shape')
                    projected=projector(feature)
                    kd=(1-F.cosine_similarity(projected,target,dim=1)).mean()
                loss=sum(losses.values())+kd
                if not torch.isfinite(loss): raise RuntimeError('Nonfinite training loss')
                group_size=min(4,len(rows)-(i//4)*4)
                (loss/group_size).backward()
                if (i+1)%4==0 or i+1==len(rows):
                    grad=torch.nn.utils.clip_grad_norm_(list(model.parameters())+list(projector.parameters()),.1)
                    if not torch.isfinite(grad): raise RuntimeError('Nonfinite gradient')
                    optimizer.step(); optimizer.zero_grad(set_to_none=True)
                total+=float(loss.detach()); kd_total+=float(kd.detach())
                status.update(status='training',epoch=epoch,step=i+1,epoch_steps=len(rows),mean_loss=total/(i+1),mean_kd=kd_total/(i+1),elapsed_seconds=time.time()-started,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20)
                if i%25==0: write_json(out/'status.json',status); print(json.dumps(status),flush=True)
                del x,targets,target,outputs,losses,loss,kd
                if mode!='none': del feature,projected
                if preflight:
                    # Runtime compatibility only; not an effectiveness experiment.
                    status.update(status='preflight_passed'); write_json(out/'status.json',status); return status
            torch.save(dict(model=model.state_dict(),projector=projector.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch,provenance=manifest,rng=dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())),out/'last.pth')
            write_json(out/'status.json',dict(status,status='validating'))
            metrics=extended_validate(model,post,dev,out,epoch)
            if epoch == start_epoch+epochs:
                from analyze_teacher_cells import analyze
                write_json(out/'teacher_cell_diagnostic.json', analyze('development', out/f'predictions_epoch_{epoch:03d}.json'))
            with (out/'history.jsonl').open('a',encoding='utf-8') as f: f.write(json.dumps(dict(metrics,mean_kd=kd_total/len(rows),elapsed_seconds=time.time()-started))+'\n')
        status.update(status='completed',metrics=metrics); write_json(out/'status.json',status)
        return status
    except Exception as e:
        status.update(status='failed',error=str(e),traceback=traceback.format_exc()); write_json(out/'status.json',status); raise
    finally:
        if hook: hook.remove()

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--mode',choices=['none','bilinear','anyup'],required=True); parser.add_argument('--name',required=True); parser.add_argument('--epochs',type=int,default=3); parser.add_argument('--preflight',action='store_true')
    args=parser.parse_args()
    if Path(args.name).name!=args.name or args.epochs<1: raise ValueError('Invalid finite run arguments')
    run(args.mode,ROOT/'results'/args.name,args.epochs,args.preflight)
