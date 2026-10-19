"""Frozen CLIP patch features and image-guided AnyUp object references."""
from pathlib import Path
import sys
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.transforms import functional as TF
from .losses import normalize, region_pool


class Teacher(nn.Module):
    def __init__(self, clip_checkpoint, anyup_checkpoint, bilinear=False):
        super().__init__()
        import clip
        model, _ = clip.load(str(clip_checkpoint), device='cpu', jit=False)
        self.visual = model.visual.float()
        self.bilinear = bilinear
        if not bilinear:
            sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'third_party/anyup'))
            from anyup.model import AnyUp
            self.upsampler = AnyUp(use_natten=False)
            self.upsampler.load_state_dict(torch.load(anyup_checkpoint, map_location='cpu', weights_only=True))
        self.register_buffer('clip_mean', torch.tensor([.48145466, .4578275, .40821073])[None, :, None, None])
        self.register_buffer('clip_std', torch.tensor([.26862954, .26130258, .27577711])[None, :, None, None])
        self.register_buffer('rgb_mean', torch.tensor([.485, .456, .406])[None, :, None, None])
        self.register_buffer('rgb_std', torch.tensor([.229, .224, .225])[None, :, None, None])
        self.eval().requires_grad_(False)

    def patch_features(self, image):
        v = self.visual
        x = v.conv1((image-self.clip_mean)/self.clip_std)
        b, channels, height, width = x.shape
        x = x.reshape(b, channels, -1).permute(0, 2, 1)
        cls = v.class_embedding[None, None].expand(b, 1, -1)
        position = v.positional_embedding
        side = int((len(position)-1)**.5)
        spatial = position[1:].T.reshape(1, channels, side, side)
        spatial = F.interpolate(spatial, (height, width), mode='bicubic', align_corners=False)
        position = torch.cat([position[:1], spatial.flatten(2).transpose(1, 2)[0]], dim=0)
        x = v.ln_pre(torch.cat([cls, x], dim=1)+position[None])
        x = v.transformer(x.permute(1, 0, 2)).permute(1, 0, 2)
        # All final-layer patches receive ln_post; visual.proj is never applied.
        return v.ln_post(x[:, 1:]).transpose(1, 2).reshape(b, channels, height, width).float()

    @torch.no_grad()
    def forward(self, images, targets, generator=None, views=True, dense=False):
        result = []
        for image, target in zip(images, targets):
            image = image[None].float()
            references, clean, fine_clean = [], None, None
            for view in range(3 if views else 1):
                x = image
                if view:
                    factors = .9+.2*torch.rand(2, generator=generator)
                    x = TF.adjust_contrast(TF.adjust_brightness(x, float(factors[0])), float(factors[1]))
                feature = self.patch_features(x)
                if self.bilinear:
                    fine = F.interpolate(feature, (160, 160), mode='bilinear', align_corners=False)
                else:
                    fine = self.upsampler((x-self.rgb_mean)/self.rgb_std, feature,
                                           output_size=(160, 160), q_chunk_size=64)
                references.append(normalize(region_pool(fine, target['boxes'])).detach())
                if view == 0:
                    clean = feature.detach()
                    if dense:
                        fine_clean = fine.detach()
            result.append({'coarse': clean, 'objects': references[0],
                           'views': references[1:], 'dense': fine_clean})
        return result
