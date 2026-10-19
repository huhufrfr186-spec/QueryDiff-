"""Full-split training with AMP, gradient accumulation and distributed reduction."""
import contextlib
import json
import math
import os
from pathlib import Path
import random
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from .data import VisDrone, collate
from .losses import graph_edges, object_losses, deep_loss, region_pool
from .models import Student
from .teacher import Teacher
from .optimization import gam_share, gam_update, GradientController, lmn_coefficients, average, SplitGradientAccumulator
from .predictions import decode
from .evaluation import qde_evaluate, official_visdrone


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_targets(targets, device):
    return [{k: v.to(device) if torch.is_tensor(v) else v for k, v in t.items()} for t in targets]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def choose_edges(boxes, mode, generator, strategy='shared'):
    shared = graph_edges(boxes)
    if len(boxes) < 2:
        return shared
    pairs = torch.triu_indices(len(boxes), len(boxes), 1, device=boxes.device).T
    if mode == 'A3':
        return pairs
    if mode == 'A4':
        return pairs[torch.randperm(len(pairs), device=boxes.device, generator=generator)[:len(shared)]]
    if strategy == 'shared' or not len(shared):
        return shared
    distances = ((boxes[pairs[:, 0], :2]-boxes[pairs[:, 1], :2])*640).norm(dim=-1)
    if strategy == 'nearest':
        return pairs[torch.argsort(distances, stable=True)[:len(shared)]]
    shared_distances = ((boxes[shared[:, 0], :2]-boxes[shared[:, 1], :2])*640).norm(dim=-1)
    bins = torch.floor(distances/2).long()
    reference_bins = torch.floor(shared_distances/2).long()
    chosen = []
    for value in reference_bins.unique(sorted=True):
        available = torch.where(bins == value)[0]
        count = int((reference_bins == value).sum())
        chosen.append(available[torch.randperm(len(available), device=boxes.device, generator=generator)[:count]])
    return pairs[torch.cat(chosen)]


@torch.no_grad()
def validate(student, dataset, device):
    student.eval()
    records = []
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate)
    for images, targets in loader:
        records.extend(decode(student(images.to(device)), targets))
    return records


def run(config, output, device='cuda', resume=None):
    world = int(os.environ.get('WORLD_SIZE', 1)); rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    if world > 1:
        dist.init_process_group('nccl' if os.name != 'nt' and device == 'cuda' else 'gloo')
    if device == 'cuda':
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda', local_rank)
    else:
        device = torch.device(device)
    seed = int(config['seed'])
    seed_all(seed)
    torch.backends.cudnn.benchmark = False
    train = VisDrone(config['train_data'], train=True, strong=config.get('strong_augmentation', False))
    validation = VisDrone(config['val_data']) if rank == 0 else None
    sampler = DistributedSampler(train, shuffle=True, seed=seed) if world > 1 else None
    microbatch = int(config['microbatch'])
    effective = int(config['effective_batch'])
    if effective % (world*microbatch):
        raise ValueError('effective_batch must be divisible by world_size * microbatch')
    accumulation = effective//(world*microbatch)
    loader = DataLoader(train, batch_size=microbatch, sampler=sampler, shuffle=sampler is None,
                        num_workers=config.get('workers', 0), collate_fn=collate,
                        generator=torch.Generator().manual_seed(seed))
    student = Student(config['initialization']).to(device)
    teacher = Teacher(config['clip_checkpoint'], config['anyup_checkpoint'],
                       bilinear=config['mode'] == 'A9').to(device)
    if world > 1:
        for p in student.parameters():
            dist.broadcast(p.data, 0)
    backbone, remaining = [], []
    for name, p in student.named_parameters():
        if p.requires_grad:
            (backbone if name.startswith('detector.backbone.') else remaining).append(p)
    optimizer = torch.optim.AdamW([{'params': backbone, 'lr': config['backbone_lr']},
                                  {'params': remaining, 'lr': config['lr']}],
                                 betas=(.9, .999), weight_decay=1e-4)
    initial_lrs = [config['backbone_lr'], config['lr']]
    amp = bool(config.get('amp', True) and device.type == 'cuda')
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    scaler.scale(torch.zeros((), device=device))
    control = GradientController(config.get('gradient_target', .1))
    split_gradients = SplitGradientAccumulator(student) if config.get('gradient_control', False) else None
    deep_weight = 20.; start_epoch = update = 0
    output = Path(output)
    if rank == 0:
        output.mkdir(parents=True, exist_ok=resume is not None)
    if world > 1:
        dist.barrier()
    if rank == 0:
        write_json(output/'config.json', config)
    if resume:
        checkpoint = torch.load(resume, map_location='cpu', weights_only=False)
        for name in ['detector', 'query_projection', 'deep_projection']:
            getattr(student, name).load_state_dict(checkpoint[name])
        optimizer.load_state_dict(checkpoint['optimizer']); scaler.load_state_dict(checkpoint['scaler'])
        start_epoch, update, deep_weight = checkpoint['epoch'], checkpoint['update'], checkpoint['deep_weight']
        control.__dict__.update(checkpoint.get('controller', {}))
    seed_all(seed+rank)
    view_generator = torch.Generator().manual_seed(seed+100000*rank)
    total_updates = math.ceil(len(loader)/accumulation)*config['epochs']
    for epoch in range(start_epoch, config['epochs']):
        student.train(); teacher.eval(); train.epoch = epoch
        if sampler is not None:
            sampler.set_epoch(epoch)
        # Independent epoch streams preserve paired data order on resumed runs.
        seed_all(seed+rank+epoch*1000003)
        loader.generator.manual_seed(seed+epoch*1000003)
        view_generator.manual_seed(seed+rank*100000+epoch*1000003)
        optimizer.zero_grad(set_to_none=True)
        ratios, records, invalid_group = [], [], False
        for step, (images, targets) in enumerate(loader):
            images = images.to(device); targets = move_targets(targets, device)
            if world > 1:
                # Match data-parallel buffer semantics while allowing independent
                # autograd.grad measurements for the auxiliary controllers.
                for buffer in student.buffers():
                    dist.broadcast(buffer, 0)
            reference = teacher(images, targets, generator=view_generator)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                outputs, projected_queries, projected_deep, region = student(images, targets)
                detection = sum(student.criterion(outputs, targets).values()).float()
            with torch.autocast(device_type=device.type, enabled=False):
                semantic = deep_loss(projected_deep, torch.cat([r['coarse'] for r in reference]))
                indices = student.criterion.matcher({k: outputs[k].float() for k in ['pred_logits', 'pred_boxes']}, targets)['indices']
                anchors, relations = [], []
                for bi, ((qi, gi), meta, ref) in enumerate(zip(indices, targets, reference)):
                    qi, gi = qi.to(device), gi.to(device)
                    order = torch.argsort(gi, stable=True)
                    qi, gi = qi[order], gi[order]
                    boxes = meta['boxes'][gi]
                    z = projected_queries[bi, qi].float()
                    if config['mode'] in ['R', 'C1']:
                        z = student.query_projection(region_pool(region[bi:bi+1].float(), boxes))
                    generator = torch.Generator(device=device).manual_seed(seed+epoch*1000003+int(meta['image_id']))
                    edges = choose_edges(boxes, config['mode'], generator, config.get('graph', 'shared'))
                    form = {'A2': 'uniform', 'A10': 'zero_variation'}.get(config['mode'], config.get('weight_form', 'full'))
                    anchor, relation, _ = object_losses(z, ref['objects'][gi], [v[gi] for v in ref['views']], edges,
                        form=form, degree=config['mode'] == 'degree', mean_normalize=config.get('mean_normalize', False),
                        global_energy=config.get('global_energy') if config['mode'] == 'A11' else None,
                        error={'A5': 'length', 'A6': 'cosine', 'A7': 'shuffle'}.get(config['mode'], 'vector'), generator=generator)
                    anchors.append(anchor); relations.append(relation)
                anchor, relation = torch.stack(anchors).mean(), torch.stack(relations).mean()
                mode = config['mode']
                la = 0. if mode in ['B', 'A8'] else config['anchor_weight']
                lr = 0. if mode in ['B', 'Q', 'R', 'A1'] else config['relation_weight']
                base = detection+(deep_weight if config.get('gam', True) else 20.)*semantic+la*anchor
                try:
                    if config.get('lmn', False):
                        aifi = [p for n, p in student.named_parameters() if n.startswith('detector.encoder.encoder.') and p.requires_grad]
                        c1, c2 = lmn_coefficients(detection, 20*semantic, aifi)
                        base = c1*detection+c2*20*semantic+la*anchor
                    loss = base+lr*relation
                    finite = torch.isfinite(loss).all().to(torch.int32)
                except FloatingPointError:
                    loss = base+0*relation
                    finite = torch.zeros((), dtype=torch.int32, device=device)
                if world > 1:
                    dist.all_reduce(finite, op=dist.ReduceOp.MIN)
                group_size = min(accumulation, len(loader)-(step//accumulation)*accumulation)
                if bool(finite):
                    if split_gradients is not None:
                        split_gradients.add(base, relation, scaler, group_size)
                    else:
                        scaler.scale(loss/group_size).backward()
                else:
                    invalid_group = True
            if (step+1) % accumulation == 0 or step+1 == len(loader):
                if split_gradients is not None:
                    controller_finite = all(torch.isfinite(g).all() for g in split_gradients.base+split_gradients.aux)
                    valid_control = torch.tensor(int(controller_finite and not invalid_group), device=device)
                    if world > 1:
                        dist.all_reduce(valid_control, op=dist.ReduceOp.MIN)
                    if bool(valid_control):
                        lr = split_gradients.finish(control, scaler)
                    else:
                        invalid_group = True
                        for p in split_gradients.parameters:
                            p.grad = torch.zeros_like(p)
                        split_gradients.reset()
                if not any(p.grad is not None for p in student.parameters()):
                    for p in student.parameters():
                        if p.requires_grad:
                            p.grad = torch.zeros_like(p)
                scaler.unscale_(optimizer)
                finite = torch.tensor(int(not invalid_group and all(torch.isfinite(p.grad).all()
                                      for p in student.parameters() if p.grad is not None)), device=device)
                if world > 1:
                    dist.all_reduce(finite, op=dist.ReduceOp.MIN)
                if bool(finite):
                    # Average unscaled FP32 gradients before measuring GAM.
                    for p in student.parameters():
                        if p.requires_grad:
                            if p.grad is None:
                                p.grad = torch.zeros_like(p)
                            average(p.grad)
                    share = gam_share(student)
                    if share is not None:
                        ratios.append(share)
                    warmup = config['warmup_steps']
                    fraction = (update-warmup)/max(1, total_updates-warmup)
                    factor = (update+1)/max(1, warmup) if update < warmup else .01+.99*(1+math.cos(math.pi*min(1., fraction)))/2
                    for group, initial in zip(optimizer.param_groups, initial_lrs):
                        group['lr'] = initial*factor
                    torch.nn.utils.clip_grad_norm_(student.parameters(), .1)
                    scaler.step(optimizer)
                    scaler.update()
                    update += 1
                elif amp:
                    scaler.update(new_scale=max(1., scaler.get_scale()/2))
                optimizer.zero_grad(set_to_none=True); invalid_group = False
            if bool(finite):
                records.append([float(detection.detach()), float(semantic.detach()),
                                float(anchor.detach()), float(relation.detach())])
            if rank == 0 and (step+1) % 50 == 0:
                print(json.dumps({'epoch': epoch+1, 'step': step+1, 'steps': len(loader),
                                  'loss': float(loss.detach()), 'deep_weight': deep_weight}), flush=True)
        if config.get('gam', True) and not config.get('lmn', False):
            deep_weight = gam_update(deep_weight, sum(ratios)/len(ratios) if ratios else None)
        if rank == 0:
            checkpoint = {name: getattr(student, name).state_dict() for name in ['detector', 'query_projection', 'deep_projection']}
            checkpoint.update(optimizer=optimizer.state_dict(), scaler=scaler.state_dict(), epoch=epoch+1,
                              update=update, deep_weight=deep_weight, config=config, controller=control.__dict__)
            temporary = output/'last.pth.tmp'; torch.save(checkpoint, temporary); temporary.replace(output/'last.pth')
            history = {'epoch': epoch+1, 'deep_weight': deep_weight,
                       'mean_losses': np.mean(records, axis=0).tolist() if records else []}
            if (epoch+1) % config.get('eval_every', 1) == 0 or epoch+1 == config['epochs']:
                predictions = validate(student, validation, device)
                write_json(output/'predictions.json', predictions)
                history['qde'] = qde_evaluate(validation.root, predictions)
                if epoch+1 == config['epochs']:
                    history['official'] = official_visdrone(validation.root, predictions)
            with (output/'history.jsonl').open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(history)+'\n')
            print(json.dumps(history), flush=True)
        if world > 1:
            dist.barrier()
    if world > 1:
        dist.destroy_process_group()
