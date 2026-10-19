from pathlib import Path
import argparse
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import yaml
from torch.utils.data import DataLoader
from querydiff.teacher import Teacher
from querydiff.data import VisDrone, collate
from querydiff.losses import graph_edges
from querydiff.training import move_targets, write_json, seed_all


def main():
    p = argparse.ArgumentParser(description='Compute the shared A11 teacher-energy constant.')
    p.add_argument('--config', default='configs/visdrone.yaml')
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='cuda')
    a = p.parse_args(); seed_all(0)
    cfg = yaml.safe_load(Path(a.config).read_text())
    data = VisDrone(cfg['train_data'])
    loader = DataLoader(data, batch_size=cfg['microbatch'], shuffle=False, collate_fn=collate)
    teacher = Teacher(cfg['clip_checkpoint'], cfg['anyup_checkpoint']).to(a.device)
    energies = []
    for step, (images, targets) in enumerate(loader):
        if step == 100:
            break
        targets = move_targets(targets, a.device)
        refs = teacher(images.to(a.device), targets, views=False)
        for target, ref in zip(targets, refs):
            edges = graph_edges(target['boxes'])
            if len(edges):
                left, right = edges.T
                energies.append(float((ref['objects'][left]-ref['objects'][right]).square().sum(-1).mean()))
    if not energies:
        raise ValueError('No teacher-patch edges in calibration images')
    write_json(a.output, {'mean_energy': sum(energies)/len(energies), 'images_with_edges': len(energies), 'seed': 0})


if __name__ == '__main__':
    main()
