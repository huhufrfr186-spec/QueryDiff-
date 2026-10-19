"""VisDrone annotations and aspect-preserving centered letterboxing."""
from pathlib import Path
import random
import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter


def letterbox(image, size=640):
    w, h = image.size
    r = min(size/w, size/h)
    nw, nh = round(r*w), round(r*h)
    px, py = (size-nw)//2, (size-nh)//2
    canvas = Image.new('RGB', (size, size), (114, 114, 114))
    canvas.paste(image.resize((nw, nh), Image.Resampling.BILINEAR), (px, py))
    return canvas, (nw/w, nh/h, px, py)


def tensor_image(image):
    return torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1).float()/255


def read_annotations(path):
    rows = [[float(v) for v in line.split(',')[:8]] for line in Path(path).read_text().splitlines() if line.strip()]
    return np.asarray(rows, dtype=np.float64).reshape(-1, 8)


def valid_boxes(raw, width, height):
    b = raw[:, :4].copy()
    b[:, 2:] += b[:, :2]
    b[:, [0, 2]] = b[:, [0, 2]].clip(0, width)
    b[:, [1, 3]] = b[:, [1, 3]].clip(0, height)
    valid = (raw[:, 4] == 1) & (raw[:, 5] >= 1) & (raw[:, 5] <= 10)
    valid &= (b[:, 2] > b[:, 0]) & (b[:, 3] > b[:, 1])
    return b, valid


class VisDrone(torch.utils.data.Dataset):
    def __init__(self, directory, train=False, jitter=.2, strong=False):
        self.root = Path(directory)
        self.images = sorted((self.root/'images').glob('*.jpg'))
        if not self.images:
            raise FileNotFoundError(f'No images in {self.root / "images"}')
        self.train, self.jitter, self.strong = train, jitter, strong
        self.epoch = 0

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        path = self.images[index]
        image = Image.open(path).convert('RGB')
        w, h = image.size
        raw = read_annotations(self.root/'annotations'/path.with_suffix('.txt').name)
        boxes, valid = valid_boxes(raw, w, h)
        ids = np.flatnonzero(valid)
        boxes = boxes[valid]
        labels = raw[valid, 5].astype(np.int64)-1
        if self.train:
            if random.random() < .5:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                boxes[:, [0, 2]] = w-boxes[:, [2, 0]]
            strength = .4 if self.strong and self.epoch < 60 else self.jitter
            ops = [ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color]
            random.shuffle(ops)
            for op in ops:
                image = op(image).enhance(random.uniform(1-strength, 1+strength))
            if self.strong and self.epoch < 60 and random.random() < .1:
                image = image.filter(ImageFilter.GaussianBlur(random.uniform(.1, 2)))
        image, transform = letterbox(image)
        sx, sy, px, py = transform
        boxes[:, [0, 2]] = boxes[:, [0, 2]]*sx+px
        boxes[:, [1, 3]] = boxes[:, [1, 3]]*sy+py
        cxcy = (boxes[:, :2]+boxes[:, 2:])/2/640
        wh = (boxes[:, 2:]-boxes[:, :2])/640
        target = {'boxes': torch.tensor(np.column_stack([cxcy, wh]), dtype=torch.float32),
                  'labels': torch.tensor(labels), 'gt_ids': torch.tensor(ids),
                  'image_id': torch.tensor(index), 'file_name': path.name,
                  'orig_size': torch.tensor([h, w]), 'transform': torch.tensor(transform)}
        return tensor_image(image), target


def collate(batch):
    images, targets = zip(*batch)
    return torch.stack(images), list(targets)
