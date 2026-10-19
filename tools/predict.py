from pathlib import Path
import argparse
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from PIL import Image, ImageFilter
from querydiff.models import build_detector
from querydiff.data import letterbox, tensor_image
from querydiff.predictions import decode
from querydiff.training import write_json


def main():
    p = argparse.ArgumentParser(description='Export detector predictions in original-image coordinates.')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--images', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--condition', choices=['clean', 'down2', 'down4', 'blur1', 'blur2'], default='clean')
    a = p.parse_args()
    detector, _ = build_detector()
    checkpoint = torch.load(a.checkpoint, map_location='cpu', weights_only=True)
    detector.load_state_dict(checkpoint['detector'])
    detector.to(a.device).eval()
    path = Path(a.images)
    files = sorted(path.glob('*.jpg')) if path.is_dir() else [path]
    records = []
    for file in files:
        image = Image.open(file).convert('RGB'); w, h = image.size
        image, transform = letterbox(image)
        if a.condition.startswith('down'):
            side = 320 if a.condition == 'down2' else 160
            image = image.resize((side, side), Image.Resampling.BILINEAR).resize((640, 640), Image.Resampling.BILINEAR)
        elif a.condition.startswith('blur'):
            image = image.filter(ImageFilter.GaussianBlur(1 if a.condition == 'blur1' else 2))
        with torch.inference_mode():
            outputs = detector(tensor_image(image)[None].to(a.device))
        records.extend(decode(outputs, [{'file_name': file.name, 'orig_size': [h, w], 'transform': transform}]))
    write_json(a.output, records)


if __name__ == '__main__':
    main()
