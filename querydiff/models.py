"""RT-DETRv4-S student and independent distillation projections."""
from pathlib import Path
import sys
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'third_party/rtdetrv4'))


def build_detector(initialization=None, num_classes=10):
    from engine.core import YAMLConfig
    cfg = YAMLConfig(str(ROOT/'third_party/rtdetrv4/configs/rtv4/rtv4_hgnetv2_s_coco.yml'))
    cfg.yaml_cfg['HGNetv2']['pretrained'] = False
    detector = cfg.model
    if initialization:
        state = torch.load(initialization, map_location='cpu', weights_only=True)
        if 'ema' in state:
            state = state['ema']['module']
        elif 'model' in state:
            state = state['model']
        detector.load_state_dict(state, strict=True)
    decoder = detector.decoder
    decoder.enc_score_head = nn.Linear(decoder.enc_score_head.in_features, num_classes)
    decoder.dec_score_head = nn.ModuleList([nn.Linear(h.in_features, num_classes)
                                           for h in decoder.dec_score_head])
    for head in [decoder.enc_score_head, *decoder.dec_score_head]:
        nn.init.constant_(head.bias, -4.59511985)
    decoder.denoising_class_embed = nn.Embedding(num_classes+1,
        decoder.denoising_class_embed.embedding_dim, padding_idx=num_classes)
    decoder.num_classes = num_classes
    detector.encoder.feature_projector = None
    for module in detector.modules():
        for key, value in list(vars(module).items()):
            if key.startswith('pos_embed') and isinstance(value, torch.Tensor):
                delattr(module, key)
                module.register_buffer(key, value, persistent=False)
    cfg.yaml_cfg['num_classes'] = num_classes
    cfg.yaml_cfg['remap_mscoco_category'] = False
    settings = cfg.yaml_cfg['RTv4Criterion']
    settings['losses'] = [v for v in settings['losses'] if v != 'distill']
    settings['weight_dict'].pop('loss_distill', None)
    settings['distill_adaptive_params']['enabled'] = False
    return detector, cfg.criterion


class Student(nn.Module):
    def __init__(self, initialization=None, num_classes=10):
        super().__init__()
        self.detector, self.criterion = build_detector(initialization, num_classes)
        self.query_projection = nn.Linear(256, 768, bias=False)
        self.deep_projection = nn.Conv2d(256, 768, 1, bias=False)
        nn.init.xavier_uniform_(self.query_projection.weight)
        nn.init.xavier_uniform_(self.deep_projection.weight)
        self._features = {}
        self.detector.encoder.register_forward_hook(self._capture_region)
        self.detector.encoder.encoder[0].register_forward_hook(self._capture_deep)
        self.detector.decoder.dec_score_head[-1].register_forward_pre_hook(self._capture_queries)

    def _capture_region(self, module, inputs, output):
        if self.training:
            self._features['region'] = output[0][0] if isinstance(output, tuple) else output[0]

    def _capture_deep(self, module, inputs, output):
        if self.training:
            b, n, channels = output.shape
            side = int(n**.5)
            self._features['deep'] = output.transpose(1, 2).reshape(b, channels, side, side)

    def _capture_queries(self, module, inputs):
        if self.training:
            self._features['queries'] = inputs[0][:, -self.detector.decoder.num_queries:]

    def forward(self, images, targets=None):
        outputs = self.detector(images, targets)
        if self.training:
            return outputs, self.query_projection(self._features.pop('queries').float()), self.deep_projection(self._features.pop('deep')), self._features.pop('region')
        return outputs
