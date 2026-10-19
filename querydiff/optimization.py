"""GAM and auxiliary-gradient strength control."""
import torch
import torch.distributed as dist


def average(tensor):
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(tensor)
        tensor /= dist.get_world_size()
    return tensor


def parameter_groups(module):
    groups = {k: [] for k in ['backbone', 'aifi', 'encoder', 'decoder', 'projections']}
    for name, p in module.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith('detector.backbone.'):
            key = 'backbone'
        elif name.startswith('detector.encoder.encoder.'):
            key = 'aifi'
        elif name.startswith('detector.encoder.'):
            key = 'encoder'
        elif name.startswith('detector.decoder.'):
            key = 'decoder'
        else:
            key = 'projections'
        groups[key].append(p)
    return groups


def gam_share(module):
    groups = parameter_groups(module)
    norms = {name: sum(float(p.grad.detach().float().abs().sum())
                       for p in params if p.grad is not None) for name, params in groups.items()}
    total = sum(norms.values())
    return norms['aifi'] / total if total > 1e-12 else None


def gam_update(weight, ratio):
    if ratio is None:
        return weight
    if ratio > .12:
        weight *= .10 / ratio
    elif ratio < .10:
        weight *= .12 / max(ratio, 1e-12)
    return min(1e4, max(1e-4, weight))


class GradientController:
    """EMA L2 gradients on deployed detector parameters before norm clipping."""
    def __init__(self, target=.1, decay=.95):
        self.target, self.decay = target, decay
        self.base_ema = self.aux_ema = None

    def coefficient(self, base, aux, parameters):
        parameters = [p for p in parameters if p.requires_grad]
        norms = []
        for loss in [base, aux]:
            gradients = torch.autograd.grad(loss.float(), parameters, retain_graph=True, allow_unused=True)
            squared = loss.new_zeros((), dtype=torch.float32)
            for p, g in zip(parameters, gradients):
                value = torch.zeros_like(p) if g is None else g
                squared += average(value.detach().float().clone()).square().sum()
            norms.append(float(squared.sqrt()))
        return self.from_norms(*norms)

    def from_norms(self, g0, ga):
        norms = [g0, ga]
        if not all(torch.isfinite(torch.tensor(norms))):
            raise FloatingPointError('Nonfinite auxiliary-controller gradients')
        g0, ga = norms
        if g0 <= 1e-12 or ga <= 1e-12:
            return 0.
        self.base_ema = g0 if self.base_ema is None else self.decay*self.base_ema+(1-self.decay)*g0
        self.aux_ema = ga if self.aux_ema is None else self.decay*self.aux_ema+(1-self.decay)*ga
        return min(1e3, max(1e-4, self.target*self.base_ema/max(self.aux_ema, 1e-12)))


class SplitGradientAccumulator:
    def __init__(self, student):
        self.parameters = [p for p in student.parameters() if p.requires_grad]
        detector_ids = {id(p) for p in student.detector.parameters()}
        self.deployed = [id(p) in detector_ids for p in self.parameters]
        self.reset()

    def reset(self):
        self.base = [torch.zeros_like(p, dtype=torch.float32) for p in self.parameters]
        self.aux = [torch.zeros_like(p, dtype=torch.float32) for p in self.parameters]

    def add(self, base, aux, scaler, divisor):
        scale = scaler.get_scale()
        for loss, storage, retain in [(base, self.base, True), (aux, self.aux, False)]:
            gradients = torch.autograd.grad(scaler.scale(loss), self.parameters,
                                             retain_graph=retain, allow_unused=True)
            for value, g in zip(storage, gradients):
                if g is not None:
                    value.add_(g.detach().float(), alpha=1/(scale*divisor))

    def finish(self, controller, scaler):
        norms = []
        for storage in [self.base, self.aux]:
            for value in storage:
                average(value)
            squared = sum(value.square().sum() for value, deployed in zip(storage, self.deployed) if deployed)
            norms.append(float(squared.sqrt()))
        coefficient = controller.from_norms(*norms)
        for p, gb, ga in zip(self.parameters, self.base, self.aux):
            p.grad = ((gb+coefficient*ga)*scaler.get_scale()).to(p.dtype)
        self.reset()
        return coefficient


def lmn_coefficients(detection, semantic, parameters, epsilon=1e-6):
    gradients = []
    for loss in [detection, semantic]:
        gs = torch.autograd.grad(torch.log(loss.float()+epsilon), parameters,
                                 retain_graph=True, allow_unused=True)
        gradients.append(torch.cat([average((torch.zeros_like(p) if g is None else g)
                                             .detach().float().clone()).flatten()
                                    for p, g in zip(parameters, gs)]))
    u, v = gradients
    if not torch.isfinite(u).all() or not torch.isfinite(v).all():
        raise FloatingPointError('Nonfinite LMN gradients')
    denominator = (u-v).square().sum()
    alpha = ((v*(v-u)).sum()/denominator).clamp(.05, .95) if denominator > 1e-12 else u.new_tensor(.5)
    r1 = alpha/(detection.detach().float()+epsilon)
    r2 = (1-alpha)/(semantic.detach().float()+epsilon)
    return (2*r1/(r1+r2)).detach(), (2*r2/(r1+r2)).detach()
