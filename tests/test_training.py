from pathlib import Path
import pytest
import torch
from torch import nn
import querydiff.training as training


class TinyData(torch.utils.data.Dataset):
    def __init__(self, root, **kwargs):
        self.root = Path(root)
        self.epoch = 0

    def __len__(self):
        return 3

    def __getitem__(self, index):
        return torch.ones(3, 8, 8), {'boxes': torch.tensor([[.01, .01, .01, .01], [.02, .02, .01, .01]]),
            'labels': torch.zeros(2, dtype=torch.long), 'image_id': torch.tensor(index),
            'file_name': f'{index}.jpg', 'orig_size': torch.tensor([8, 8]),
            'transform': torch.tensor([80., 80., 0., 0.])}


class TinyCriterion(nn.Module):
    def forward(self, outputs, targets):
        return {'loss': outputs['pred_logits'].square().mean()+outputs['pred_boxes'].square().mean()}

    def matcher(self, outputs, targets):
        return {'indices': [(torch.arange(2), torch.arange(2)) for _ in targets]}


class TinyStudent(nn.Module):
    def __init__(self, initialization):
        super().__init__()
        self.detector = nn.Module()
        self.detector.backbone = nn.Conv2d(3, 4, 1)
        self.detector.encoder = nn.Module()
        self.detector.encoder.encoder = nn.Linear(4, 4)
        self.detector.decoder = nn.Linear(4, 14)
        self.query_projection = nn.Linear(4, 768, bias=False)
        self.deep_projection = nn.Conv2d(4, 768, 1, bias=False)
        self.criterion = TinyCriterion()

    def forward(self, images, targets):
        region = self.detector.backbone(images)
        vector = region.mean((-2, -1))
        queries = self.detector.encoder.encoder(torch.stack([vector, .7*vector], dim=1))
        output = self.detector.decoder(queries)
        predictions = {'pred_logits': output[..., :10], 'pred_boxes': output[..., 10:].sigmoid()}
        return predictions, self.query_projection(queries), self.deep_projection(region), region


class TinyTeacher(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, images, targets, **kwargs):
        objects = torch.eye(2, 768, device=images.device)
        return [{'coarse': torch.ones(1, 768, 2, 2, device=images.device),
                 'objects': objects, 'views': [objects, objects]} for _ in targets]


@pytest.mark.parametrize('branch', ['gam', 'strength', 'lmn'])
def test_training_accumulation_and_checkpoint(tmp_path, monkeypatch, branch):
    monkeypatch.setattr(training, 'VisDrone', TinyData)
    monkeypatch.setattr(training, 'Student', TinyStudent)
    monkeypatch.setattr(training, 'Teacher', TinyTeacher)
    monkeypatch.setattr(training, 'validate', lambda *args: [])
    monkeypatch.setattr(training, 'qde_evaluate', lambda *args: {})
    monkeypatch.setattr(training, 'official_visdrone', lambda *args: {})
    config = {'seed': 0, 'train_data': 'train', 'val_data': 'val', 'microbatch': 1,
        'effective_batch': 2, 'initialization': None, 'clip_checkpoint': None,
        'anyup_checkpoint': None, 'mode': 'M', 'lr': 4e-4, 'backbone_lr': 2e-4,
        'epochs': 1, 'warmup_steps': 2, 'anchor_weight': 1., 'relation_weight': .1,
        'gam': branch == 'gam', 'gradient_control': branch == 'strength',
        'lmn': branch == 'lmn', 'mean_normalize': branch == 'strength'}
    output = tmp_path/'run'
    training.run(config, output, device='cpu')
    saved = torch.load(output/'last.pth', weights_only=True)
    assert saved['epoch'] == 1 and saved['update'] == 2
    assert saved['deep_weight'] > 0
    assert all(torch.isfinite(v).all() for v in saved['detector'].values())
    if branch == 'strength':
        assert saved['controller']['base_ema'] is not None
    config['epochs'] = 2
    training.run(config, output, device='cpu', resume=output/'last.pth')
    resumed = torch.load(output/'last.pth', weights_only=True)
    assert resumed['epoch'] == 2 and resumed['update'] == 4
