import numpy as np
import torch
from PIL import Image
from querydiff.losses import graph_edges, edge_weights, object_losses, region_pool, normalize, component_scatter
from querydiff.optimization import gam_update, gam_share, parameter_groups, GradientController, lmn_coefficients
from querydiff.data import letterbox
from querydiff.predictions import decode


def test_single_teacher_patch_graph():
    boxes = torch.tensor([[.002, .002, .01, .01], [.023, .023, .01, .01], [.026, .026, .01, .01]])
    assert graph_edges(boxes).tolist() == [[0, 1]]


def test_weight_formula_and_dimension_sum():
    t = torch.eye(3)
    views = [t*.9, t*1.1]
    edges = torch.tensor([[0, 1], [1, 2]])
    w, _, s, v, gamma = edge_weights(t, views, edges)
    assert torch.allclose(s, torch.full((2,), 2.))
    assert torch.allclose(v, torch.full((2,), .02))
    assert torch.allclose(w, s/(s+v+1e-6).square())
    assert torch.allclose(gamma, s/(s+v+1e-6))


def test_graph_quadratic_and_gradient():
    torch.manual_seed(4)
    z = torch.randn(4, 7, requires_grad=True)
    t = normalize(torch.randn(4, 7))
    edges = torch.tensor([[0, 1], [0, 2], [1, 2]])
    _, loss, stats = object_losses(z, t, [t], edges)
    incidence = torch.zeros(3, 4)
    incidence[torch.arange(3), edges[:, 0]] = 1
    incidence[torch.arange(3), edges[:, 1]] = -1
    laplacian = incidence.T@torch.diag(stats['omega']/3)@incidence
    error = z-t
    expected = torch.trace(error.T@laplacian@error)
    assert torch.allclose(loss, expected)
    gradient = torch.autograd.grad(loss, z)[0]
    assert torch.allclose(gradient, 2*laplacian@error, atol=1e-6)


def test_degree_preserves_diagonal_hessian():
    t = normalize(torch.tensor([[1., 2.], [3., 1.], [1., 4.]]))
    edges = torch.tensor([[0, 1], [1, 2]])
    z = torch.randn(3, 2, requires_grad=True)
    diff = lambda x: object_losses(x, t, [], edges)[1]
    degree = lambda x: object_losses(x, t, [], edges, degree=True)[1]
    hd = torch.autograd.functional.hessian(diff, z).reshape(6, 6)
    hg = torch.autograd.functional.hessian(degree, z).reshape(6, 6)
    assert torch.allclose(hd.diag(), hg.diag())
    assert torch.count_nonzero(hg-torch.diag(hg.diag())) == 0
    assert torch.count_nonzero(hd-torch.diag(hd.diag())) > 0


def test_identical_teacher_edge_has_zero_weight():
    t = torch.ones(2, 5)
    w, *_ = edge_weights(t, [t], torch.tensor([[0, 1]]))
    assert w.item() == 0


def test_empty_edge_differentiable_zero():
    z = torch.randn(1, 5, requires_grad=True)
    anchor, relation, _ = object_losses(z, torch.ones_like(z), [], torch.empty(0, 2, dtype=torch.long))
    assert relation.item() == 0
    (anchor+relation).backward()
    assert torch.isfinite(z.grad).all()


def test_four_by_four_continuous_pool():
    # A linear field has its exact region mean at the region center.
    yy, xx = torch.meshgrid((torch.arange(16)+.5)/16, (torch.arange(16)+.5)/16, indexing='ij')
    feature = (2*xx+3*yy)[None, None]
    boxes = torch.tensor([[.4, .6, .3, .2]])
    result = region_pool(feature, boxes)
    assert torch.allclose(result, torch.tensor([[2.6]]), atol=1e-6)


def test_component_decomposition():
    e = torch.randn(5, 9)
    result = component_scatter(e, [torch.tensor([0, 1]), torch.tensor([2, 3, 4])])
    assert torch.allclose(result['A'], result['C']+result['S'], atol=1e-6)


def test_gam_direct_ratio_update_and_projection_scope():
    assert gam_update(20, .2) == 10
    assert gam_update(20, .05) == 48
    assert gam_update(20, .11) == 20
    assert gam_update(20, None) == 20
    assert gam_update(20, 0) == 1e4
    from torch import nn
    m = nn.Module(); m.detector = nn.Module(); m.detector.encoder = nn.Module()
    m.detector.encoder.encoder = nn.Linear(1, 1, bias=False)
    m.query_projection = nn.Linear(1, 1, bias=False)
    m.detector.encoder.encoder.weight.grad = torch.ones(1, 1)
    m.query_projection.weight.grad = torch.full((1, 1), 9.)
    assert abs(gam_share(m)-.1) < 1e-8
    assert sum(len(v) for v in parameter_groups(m).values()) == 2


def test_lmn_sum_and_gradient_controller():
    p = torch.tensor([1., 2.], requires_grad=True)
    l1, l2 = p.square().sum(), (p-3).square().sum()
    c1, c2 = lmn_coefficients(l1, l2, [p])
    assert torch.allclose(c1+c2, torch.tensor(2.))
    assert not c1.requires_grad and not c2.requires_grad
    controller = GradientController(.1)
    assert abs(controller.from_norms(20., 4.)-.5) < 1e-8
    assert controller.from_norms(20., 0.) == 0


def test_letterbox_and_inverse_one_class_per_query():
    image, transform = letterbox(Image.new('RGB', (960, 540)))
    assert image.size == (640, 640)
    assert transform == (2/3, 2/3, 0, 140)
    outputs = {'pred_logits': torch.tensor([[[8., 7.], [8., 8.]]]),
               'pred_boxes': torch.tensor([[[.5, .5, .25, .25], [.25, .5, .1, .1]]])}
    result = decode(outputs, [{'transform': transform, 'orig_size': [540, 960], 'file_name': 'one.jpg'}])[0]
    assert len(result['detections']) == 2
    assert result['query_ids'] == [0, 1]
    assert np.allclose(result['detections'][0][:4], [360, 150, 240, 240])


def test_accumulated_controller_measures_effective_batch():
    from torch import nn
    from querydiff.optimization import SplitGradientAccumulator
    model = nn.Module(); model.detector = nn.Linear(1, 1, bias=False)
    model.projection = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.detector.weight.fill_(1); model.projection.weight.fill_(2)
    accumulator = SplitGradientAccumulator(model)
    scaler = torch.amp.GradScaler('cuda', enabled=False)
    for x in [torch.tensor([[1.]]), torch.tensor([[2.]])]:
        base = model.detector(x).square().mean()
        auxiliary = model.projection(model.detector(x)).square().mean()
        accumulator.add(base, auxiliary, scaler, 2)
    coefficient = accumulator.finish(GradientController(.1), scaler)
    # E[x^2]=2.5; detector base/aux gradients are 5 and 20.
    assert abs(coefficient-.025) < 1e-8
    assert torch.allclose(model.detector.weight.grad, torch.tensor([[5.5]]))
    assert torch.allclose(model.projection.weight.grad, torch.tensor([[.25]]))


def test_clip_patch_ln_post_and_interpolated_positions():
    from torch import nn
    from querydiff.teacher import Teacher
    teacher = Teacher.__new__(Teacher); nn.Module.__init__(teacher)
    visual = nn.Module(); visual.conv1 = nn.Conv2d(3, 8, 2, stride=2, bias=False)
    visual.class_embedding = nn.Parameter(torch.randn(8))
    visual.positional_embedding = nn.Parameter(torch.randn(5, 8))
    visual.ln_pre = nn.LayerNorm(8); visual.transformer = nn.Identity(); visual.ln_post = nn.LayerNorm(8)
    visual.proj = nn.Parameter(torch.randn(8, 3))
    teacher.visual = visual
    teacher.register_buffer('clip_mean', torch.zeros(1, 3, 1, 1))
    teacher.register_buffer('clip_std', torch.ones(1, 3, 1, 1))
    image = torch.rand(1, 3, 8, 8)
    result = teacher.patch_features(image)
    position = torch.nn.functional.interpolate(visual.positional_embedding[1:].T.reshape(1, 8, 2, 2),
                                                (4, 4), mode='bicubic', align_corners=False)
    expected = visual.ln_post(visual.ln_pre((visual.conv1(image)+position).flatten(2).transpose(1, 2)))
    assert result.shape == (1, 8, 4, 4)
    assert torch.allclose(result.flatten(2).transpose(1, 2), expected, atol=1e-6)
