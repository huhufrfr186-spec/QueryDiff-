"""Object anchoring, teacher-patch graphs, and local difference objectives."""
import torch
from torch.nn import functional as F


def normalize(x):
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-6)


def region_pool(feature, boxes, samples=4):
    """Mean bilinear samples at 4x4 bin centers; normalized cxcywh boxes."""
    if not len(boxes):
        return feature.new_empty((0, feature.shape[1]))
    offsets = (torch.arange(samples, device=boxes.device).float() + .5) / samples - .5
    yy, xx = torch.meshgrid(offsets, offsets, indexing='ij')
    grid = boxes[:, None, None, :2] + torch.stack((xx, yy), -1)[None] * boxes[:, None, None, 2:]
    sampled = F.grid_sample(feature.float().expand(len(boxes), -1, -1, -1),
                            grid.float() * 2 - 1, mode='bilinear',
                            padding_mode='border', align_corners=False)
    return sampled.mean((-2, -1))


def graph_edges(boxes, image_size=640, patch_size=16):
    if len(boxes) < 2:
        return torch.empty((0, 2), device=boxes.device, dtype=torch.long)
    location = boxes[:, :2].detach().float() * (image_size / patch_size)
    location = torch.where((location-location.round()).abs() < 1e-5, location.round(), location)
    cells = location.floor().long()
    cells.clamp_(0, image_size // patch_size - 1)
    pair = torch.triu_indices(len(boxes), len(boxes), 1, device=boxes.device).T
    return pair[(cells[pair[:, 0]] == cells[pair[:, 1]]).all(-1)]


def edge_weights(teacher, views, edges, form='full', epsilon=1e-6,
                 mean_normalize=False, global_energy=None):
    """s/(s+v+eps)^2 = gamma/a; retain sums over feature dimensions."""
    teacher = teacher.detach().float()
    left, right = edges.T
    delta = teacher[left] - teacher[right]
    energy = delta.square().sum(-1)
    variation = (torch.stack([((v.detach().float()[left] - v.detach().float()[right]) - delta)
                             .square().sum(-1) for v in views]).mean(0)
                 if views else torch.zeros_like(energy))
    used_energy = energy if global_energy is None else torch.full_like(energy, float(global_energy))
    if form == 'zero_variation':
        variation = torch.zeros_like(variation)
    a = used_energy + variation + epsilon
    gamma = used_energy / a
    if form == 'uniform':
        weight = torch.ones_like(energy)
    elif form == 'gamma':
        weight = gamma
    else:
        weight = gamma / a
    if mean_normalize and len(weight):
        mean = weight.mean()
        weight = weight / mean if mean > 1e-12 else torch.zeros_like(weight)
    return weight.detach(), delta.detach(), energy.detach(), variation.detach(), gamma.detach()


def object_losses(z, teacher, views, edges, form='full', degree=False,
                  mean_normalize=False, global_energy=None, error='vector', generator=None):
    z = z.float()
    teacher = teacher.detach().float()
    zero = z.sum() * 0
    anchor = (z - teacher).square().sum(-1).mean() if len(z) else zero
    if not len(edges):
        return anchor, zero, {'edges': 0}
    w, target, s, v, gamma = edge_weights(teacher, views, edges, form,
                                         mean_normalize=mean_normalize, global_energy=global_energy)
    left, right = edges.T
    if degree:
        deg = z.new_zeros(len(z)).index_add(0, left, w).index_add(0, right, w)
        relation = (deg * (z - teacher).square().sum(-1)).sum() / len(edges)
    else:
        delta = z[left] - z[right]
        if error == 'shuffle':
            target = target[torch.randperm(len(target), device=z.device, generator=generator)]
        if error == 'length':
            residual = (delta.norm(dim=-1) - target.norm(dim=-1)).square()
        elif error == 'cosine':
            residual = ((normalize(z[left]) * normalize(z[right])).sum(-1) -
                        (teacher[left] * teacher[right]).sum(-1)).square()
        else:
            residual = (delta - target).square().sum(-1)
        relation = (w * residual).mean()
    return anchor, relation, {'edges': len(edges), 's': s, 'v': v, 'gamma': gamma, 'omega': w}


def deep_loss(projected, teacher):
    teacher = F.interpolate(teacher.detach().float(), projected.shape[-2:],
                            mode='bilinear', align_corners=False)
    p = normalize(projected.float().permute(0, 2, 3, 1))
    t = normalize(teacher.permute(0, 2, 3, 1))
    return (1 - (p * t).sum(-1).clamp(-1, 1)).mean()


def component_scatter(errors, components):
    """Return object-weighted per-dimension C, S, A on fixed components."""
    denom = errors.numel()
    c = errors.new_zeros(())
    s = errors.new_zeros(())
    for indices in components:
        e = errors[indices]
        mean = e.mean(0)
        c = c + len(e) * mean.square().sum()
        s = s + (e - mean).square().sum()
    return {'C': c / max(1, denom), 'S': s / max(1, denom),
            'A': errors.square().sum() / max(1, denom)}
