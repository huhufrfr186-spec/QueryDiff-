"""Auditable, bounded VisDrone transfer baseline; not official COCO reproduction."""
from bootstrap import ROOT, result_dir, sha256, write_json
import argparse
import json
import math
import random
import time
import traceback
from collections import defaultdict
import numpy as np
import torch
from torch import nn
from PIL import Image, ImageEnhance
from torchvision.transforms.functional import pil_to_tensor
from engine.core import YAMLConfig
from faster_coco_eval import COCO, COCOeval_faster

KNOWN = [1, 2, 4, 5, 6, 9, 10]
COCO_INIT = [0, 0, 2, 2, 7, 5, 3]
CONFIG = ROOT/'work/RT-DETRv4/configs/rtv4/rtv4_hgnetv2_s_coco.yml'
WEIGHTS = ROOT/'models/rtdetrv4_s_official.pth'
DATA = ROOT/'data/research_protocol_v1/coco_split_A'
IMAGES = ROOT/'data/VisDrone2019-DET-train/images'


def build_model():
    cfg = YAMLConfig(str(CONFIG))
    cfg.yaml_cfg['HGNetv2']['pretrained'] = False
    model = cfg.model
    ckpt = torch.load(WEIGHTS, map_location='cpu', weights_only=True)
    model.load_state_dict(ckpt['ema']['module'], strict=True)
    # Coarse COCO semantics initialize all heads, but training labels are the
    # seven distinct VisDrone classes; people/van are NOT collapsed for scoring.
    def transfer(head):
        new = nn.Linear(head.in_features, len(KNOWN))
        with torch.no_grad():
            new.weight.copy_(head.weight[COCO_INIT])
            new.bias.copy_(head.bias[COCO_INIT])
        return new
    model.decoder.enc_score_head = transfer(model.decoder.enc_score_head)
    model.decoder.dec_score_head = nn.ModuleList([transfer(h) for h in model.decoder.dec_score_head])
    old = model.decoder.denoising_class_embed
    embedding = nn.Embedding(8, old.embedding_dim, padding_idx=7)
    with torch.no_grad():
        embedding.weight[:7].copy_(old.weight[COCO_INIT])
        embedding.weight[7].zero_()
    model.decoder.denoising_class_embed = embedding
    model.decoder.num_classes = 7
    # No additional teacher in baseline. Existing checkpoint already benefited
    # from COCO + DINOv3 pretraining; this is not a from-scratch no-VFM baseline.
    model.encoder.feature_projector = None
    for module in model.modules():
        for key, value in list(vars(module).items()):
            if key.startswith('pos_embed') and isinstance(value, torch.Tensor):
                delattr(module, key)
                module.register_buffer(key, value, persistent=False)
    cfg.yaml_cfg['num_classes'] = 7
    cfg.yaml_cfg['remap_mscoco_category'] = False
    settings = cfg.yaml_cfg['RTv4Criterion']
    settings['losses'] = [x for x in settings['losses'] if x != 'distill']
    settings['weight_dict'].pop('loss_distill', None)
    settings['distill_adaptive_params']['enabled'] = False
    return model, cfg.criterion, cfg.postprocessor


class Dataset:
    def __init__(self, split, limit=0):
        self.path = DATA/f'{split}.json'
        self.raw = json.loads(self.path.read_text(encoding='utf-8'))
        self.rows = sorted(self.raw['images'], key=lambda x: x['file_name'])
        if limit:
            self.rows = self.rows[:limit]
        self.annotations = defaultdict(list)
        for a in self.raw['annotations']:
            self.annotations[a['image_id']].append(a)

    def sample(self, row, augment=False):
        with Image.open(IMAGES/row['file_name']) as im:
            image = im.convert('RGB')
        flip = augment and random.random() < .5
        if flip:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if augment:
            image = ImageEnhance.Brightness(image).enhance(random.uniform(.8, 1.2))
            image = ImageEnhance.Contrast(image).enhance(random.uniform(.8, 1.2))
        image = image.resize((640, 640), Image.Resampling.BILINEAR)
        x = pil_to_tensor(image).float().div_(255).unsqueeze(0).cuda()
        b, labels = [], []
        w, h = row['width'], row['height']
        for a in self.annotations[row['id']]:
            left, top, bw, bh = a['bbox']
            # Clamp exported boxes to source bounds; reject only degenerate boxes.
            right, bottom = min(w, left+bw), min(h, top+bh)
            left, top = max(0, left), max(0, top)
            if right <= left or bottom <= top:
                continue
            cx, cy = (left+right)/(2*w), (top+bottom)/(2*h)
            b.append([1-cx if flip else cx, cy, (right-left)/w, (bottom-top)/h])
            labels.append(a['category_id'])
        target = dict(boxes=torch.tensor(b, device='cuda', dtype=torch.float32).reshape(-1, 4),
                      labels=torch.tensor(labels, device='cuda', dtype=torch.int64))
        return x, [target]


@torch.inference_mode()
def validate(model, post, dataset, output, epoch):
    model.eval()
    detections = []
    for row in dataset.rows:
        x, _ = dataset.sample(row)
        pred = post(model(x), torch.tensor([[row['width'], row['height']]], device='cuda'))[0]
        boxes = pred['boxes'].cpu().numpy()
        scores, labels = pred['scores'].cpu().tolist(), pred['labels'].cpu().tolist()
        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = box.tolist()
            if not math.isfinite(score) or not np.isfinite(box).all():
                raise RuntimeError('Nonfinite validation prediction')
            detections.append(dict(image_id=row['id'], category_id=label,
                                   bbox=[x1, y1, max(0, x2-x1), max(0, y2-y1)], score=score))
    ids = {r['id'] for r in dataset.rows}
    coco = COCO()
    coco.dataset = dict(images=dataset.rows, categories=dataset.raw['categories'],
                        annotations=[a for a in dataset.raw['annotations'] if a['image_id'] in ids], info={})
    coco.createIndex()
    evaluator = COCOeval_faster(coco, coco.loadRes(detections), iouType='bbox')
    evaluator.params.imgIds = sorted(ids)
    evaluator.params.catIds = list(range(7))
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    metrics = dict(epoch=epoch, images=len(ids), AP=float(evaluator.stats[0]),
                   AP50=float(evaluator.stats[1]), AP75=float(evaluator.stats[2]),
                   evaluation='COCO-style known-only export, not official VisDrone or open-set evaluation')
    write_json(output/f'validation_epoch_{epoch:03d}.json', metrics)
    write_json(output/f'predictions_epoch_{epoch:03d}.json', detections)
    return metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--name', required=True)
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--fit-limit', type=int, default=0)
    p.add_argument('--dev-limit', type=int, default=128)
    p.add_argument('--accumulation', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--lr', type=float, default=.0001)
    p.add_argument('--resume', type=__import__('pathlib').Path)
    args = p.parse_args()
    if args.epochs < 1 or args.accumulation < 1 or min(args.fit_limit, args.dev_limit) < 0:
        raise ValueError('Invalid bounded run settings')
    out = result_dir(args.name)
    started = time.time()
    status = dict(status='initializing', epoch=0, step=0)
    write_json(out/'status.json', status)
    try:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.set_num_threads(2)
        torch.backends.cudnn.benchmark = False
        fit, dev = Dataset('fit', args.fit_limit), Dataset('development', args.dev_limit)
        if {r['file_name'] for r in fit.rows} & {r['file_name'] for r in dev.rows}:
            raise RuntimeError('Fit/development overlap')
        model, criterion, post = build_model()
        model, criterion, post = model.cuda(), criterion.cuda(), post.cuda()
        optimizer = torch.optim.AdamW([
            dict(params=[v for n, v in model.named_parameters() if n.startswith('backbone.')], lr=args.lr*.2),
            dict(params=[v for n, v in model.named_parameters() if not n.startswith('backbone.')], lr=args.lr)
        ], weight_decay=.0001)
        # FP32 fits this 10M model at batch 1. Initial AMP diagnostic overflowed;
        # avoid silently skipped optimizer steps in the reproducible baseline.
        scaler = torch.amp.GradScaler('cuda', enabled=False)
        start_epoch = 0
        if args.resume:
            resume_path = args.resume.resolve()
            if not resume_path.is_relative_to((ROOT/'results').resolve()):
                raise ValueError('Resume only project-generated result checkpoints')
            previous = torch.load(resume_path, map_location='cpu', weights_only=False)
            if previous['provenance']['known_original_ids'] != KNOWN:
                raise ValueError('Resume label space mismatch')
            if previous['provenance']['fit_sha256'] != sha256(fit.path):
                raise ValueError('Resume fit protocol mismatch')
            if previous['provenance']['fit_images'] != len(fit.rows):
                raise ValueError('Resume fit selection size mismatch')
            if previous['provenance']['effective_batch'] != args.accumulation:
                raise ValueError('Resume accumulation mismatch')
            model.load_state_dict(previous['model'], strict=True)
            optimizer.load_state_dict(previous['optimizer'])
            scaler.load_state_dict(previous['scaler'])
            start_epoch = previous['epoch']
            random.setstate(previous['rng']['python'])
            np.random.set_state(previous['rng']['numpy'])
            torch.set_rng_state(previous['rng']['torch'])
            torch.cuda.set_rng_state_all(previous['rng']['cuda'])
            del previous
        if criterion.num_classes != 7 or post.num_classes != 7:
            raise RuntimeError('Criterion/postprocessor label configuration mismatch')
        provenance = dict(arguments={k: str(v) if isinstance(v, __import__('pathlib').Path) else v
                                     for k, v in vars(args).items()},
                          known_original_ids=KNOWN, initialization_coco_indices=COCO_INIT,
                          weights_sha256=sha256(WEIGHTS), script_sha256=sha256(__file__),
                          fit_sha256=sha256(fit.path), development_sha256=sha256(dev.path),
                          fit_images=len(fit.rows), development_images=len(dev.rows),
                          trainable_parameters=sum(v.numel() for v in model.parameters() if v.requires_grad),
                          loss_names=criterion.losses, loss_weights=criterion.weight_dict,
                          batch_size=1, effective_batch=args.accumulation, resolution=640,
                          precision='FP32 network, assignment, losses and gradients',
                          optimizer_group_lrs=[group['lr'] for group in optimizer.param_groups],
                          augmentation='horizontal flip + brightness/contrast 0.8..1.2; no mosaic/crop',
                          batch_norm='running statistics frozen, affine parameters trained (batch size 1)',
                          limitations=['Not official full training recipe or converged strong baseline yet',
                                       'Unknown targets visible but unlabeled during training',
                                       'COCO/VFM pretraining includes overlapping concepts',
                                       'Development subset used for monitoring; not untouched final testing',
                                       'Known-only COCO-style evaluation lacks original VisDrone ignore handling'])
        if args.resume:
            provenance['resume_sha256'] = sha256(args.resume)
        write_json(out/'official_source_hashes.json', {
            str(path.relative_to(ROOT)): sha256(path)
            for directory, pattern in [('engine', '*.py'), ('configs', '*.yml')]
            for path in sorted((ROOT/'work/RT-DETRv4'/directory).rglob(pattern))})
        write_json(out/'provenance.json', provenance)
        write_json(out/'selection.json', dict(fit=[r['file_name'] for r in fit.rows],
                                            development=[r['file_name'] for r in dev.rows]))
        (out/'train_visdrone_executed.py').write_bytes(__import__('pathlib').Path(__file__).read_bytes())
        best = -1.
        write_json(out/'status.json', dict(status, status='validating_initial'))
        initial = validate(model, post, dev, out, start_epoch)
        write_json(out/'initial_validation.json', initial)
        # Keep the resumed initial model if every subsequent epoch regresses.
        # Best means best across this run including its initial checkpoint.
        best = initial['AP']
        torch.save(dict(model=model.state_dict(), epoch=start_epoch, metrics=initial,
                        provenance=provenance), out/'best.pth')
        for epoch in range(start_epoch+1, start_epoch+args.epochs+1):
            model.train()
            for module in model.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.eval()
            rows = list(fit.rows)
            random.shuffle(rows)
            optimizer.zero_grad(set_to_none=True)
            losses_epoch = []
            for i, row in enumerate(rows):
                if (out/'STOP').exists():
                    raise InterruptedError('Run-local STOP requested')
                x, targets = fit.sample(row, augment=True)
                outputs = model(x, targets)
                # Disable AMP for assignment and losses. Nested tensors retain
                # gradients through the float casts.
                def float_tree(value):
                    if isinstance(value, torch.Tensor) and value.is_floating_point():
                        return value.float()
                    if isinstance(value, dict):
                        return {k: float_tree(v) for k, v in value.items()}
                    if isinstance(value, list):
                        return [float_tree(v) for v in value]
                    return value
                loss_dict = criterion(float_tree(outputs), targets)
                loss = sum(loss_dict.values())
                if not torch.isfinite(loss):
                    raise RuntimeError(f'Nonfinite loss at {row["file_name"]}')
                # Correct final short accumulation group denominator.
                group_size = min(args.accumulation, len(rows)-(i//args.accumulation)*args.accumulation)
                scaler.scale(loss/group_size).backward()
                if (i+1) % args.accumulation == 0 or i+1 == len(rows):
                    scaler.unscale_(optimizer)
                    grad = torch.nn.utils.clip_grad_norm_(model.parameters(), .1)
                    if not torch.isfinite(grad):
                        raise RuntimeError('Nonfinite gradients')
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                losses_epoch.append(float(loss.detach()))
                del outputs, loss_dict, loss, x, targets
                if i % 25 == 0 or i+1 == len(rows):
                    status.update(status='training', epoch=epoch, step=i+1, epoch_steps=len(rows),
                                  elapsed_seconds=time.time()-started, mean_loss=float(np.mean(losses_epoch)),
                                  peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20)
                    write_json(out/'status.json', status)
                    print(json.dumps(status), flush=True)
            # Save before validation so evaluator failure cannot discard training.
            checkpoint = dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                              scaler=scaler.state_dict(), epoch=epoch, provenance=provenance,
                              rng=dict(python=random.getstate(), numpy=np.random.get_state(),
                                       torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all()))
            torch.save(checkpoint, out/'last.pth')
            write_json(out/'status.json', dict(status, status='validating'))
            metrics = validate(model, post, dev, out, epoch)
            if metrics['AP'] > best:
                best = metrics['AP']
                torch.save(dict(model=model.state_dict(), epoch=epoch, metrics=metrics, provenance=provenance), out/'best.pth')
            with (out/'history.jsonl').open('a', encoding='utf-8') as f:
                f.write(json.dumps(dict(metrics, mean_loss=float(np.mean(losses_epoch))))+'\n')
        status.update(status='completed', best_development_AP=best, elapsed_seconds=time.time()-started)
        write_json(out/'status.json', status)
    except Exception as error:
        status.update(status='failed', error=str(error), traceback=traceback.format_exc(),
                      elapsed_seconds=time.time()-started)
        write_json(out/'status.json', status)
        raise


if __name__ == '__main__':
    main()
