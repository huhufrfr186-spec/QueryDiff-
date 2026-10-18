"""Paper equations: matched ordinary queries, bounded local edge supervision."""
import torch
from torch.nn import functional as F


def object_targets(feature, boxes, samples=3):
    """Bilinear continuous-region quadrature, coordinates on the detector image."""
    if len(boxes) == 0:
        return feature.new_empty((0, feature.shape[1]))
    offsets = (torch.arange(samples, device=boxes.device, dtype=boxes.dtype)+.5)/samples-.5
    yy, xx = torch.meshgrid(offsets, offsets, indexing='ij')
    grid = boxes[:, None, None, :2]+torch.stack((xx, yy), -1)[None]*boxes[:, None, None, 2:]
    sampled = F.grid_sample(feature.expand(len(boxes), -1, -1, -1), grid*2-1,
                            mode='bilinear', padding_mode='border', align_corners=False)
    return F.normalize(sampled.mean((-1, -2)), dim=-1, eps=1e-6).detach()


def graph_edges(boxes, image_hw=(640, 640), student_hw=(80, 80), teacher_hw=(40, 40)):
    n = len(boxes)
    if n < 2:
        return torch.empty((0, 2), device=boxes.device, dtype=torch.long)
    pair = torch.triu_indices(n, n, 1, device=boxes.device).T
    mask = torch.zeros(len(pair), device=boxes.device, dtype=torch.bool)
    for height, width in (student_hw, teacher_hw):
        cells = (boxes[:, :2]*boxes.new_tensor([width, height])).floor().long()
        cells[:, 0].clamp_(0, width-1); cells[:, 1].clamp_(0, height-1)
        mask |= (cells[pair[:, 0]] == cells[pair[:, 1]]).all(-1)
    return pair[mask]


def losses(z, teacher, perturbed, edges, mode='M', epsilon=1e-4):
    zero = z.sum()*0
    if len(z) == 0:
        return zero, zero, {'objects': 0, 'edges': 0}
    teacher = teacher.detach()
    anchor = (z-teacher).square().sum(-1).mean()
    if len(edges) == 0 or mode in ('B', 'Q'):
        return anchor, zero, {'objects': len(z), 'edges': len(edges)}
    left, right = edges.T
    target_delta = teacher[left]-teacher[right]
    energy = target_delta.square().sum(-1)
    variation = torch.stack([((t[left]-t[right])-target_delta).square().sum(-1)
                             for t in perturbed]).mean(0) if perturbed else torch.zeros_like(energy)
    denominator = energy+variation+epsilon
    coefficient = (energy/denominator.square()).detach() if mode == 'M' else torch.ones_like(energy)
    residual = (z[left]-z[right])-target_delta
    difference = (coefficient*residual.square().sum(-1)).mean()
    return anchor, difference, {'objects': len(z), 'edges': len(edges),
        'teacher_energy': float(energy.mean()), 'variation': float(variation.mean()),
        'coefficient': float(coefficient.mean()), 'zero_energy_edges': int((energy == 0).sum())}


def gam_ratio(model):
    total = target = 0.
    for name, p in model.named_parameters():
        if p.grad is not None:
            magnitude = float(p.grad.detach().abs().sum())
            total += magnitude
            if name.startswith('encoder.encoder.'):
                target += magnitude
    return 100*target/max(total, 1e-30)


def gam_update(weight, ratio, rho=11., delta=1.):
    # Official released implementation: gradient share odds ratio, bounded x10.
    if ratio <= 1e-6:
        return weight
    if rho-delta <= ratio <= rho+delta:
        return weight
    target = rho-delta if ratio > rho+delta else rho+delta
    current, desired = ratio/100., target/100.
    factor = desired*(1-current)/max(current*(1-desired), 1e-9)
    return weight*min(10.,max(.1,factor))
