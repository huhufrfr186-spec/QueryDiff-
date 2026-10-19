from pathlib import Path
import argparse
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import yaml


def main():
    p = argparse.ArgumentParser(description='Train QueryDiff on the full VisDrone training split.')
    p.add_argument('--config', default='configs/visdrone.yaml')
    p.add_argument('--output', required=True)
    p.add_argument('--mode', choices=['B', 'Q', 'M', 'degree', 'R', 'C1']+[f'A{i}' for i in range(1, 12)])
    p.add_argument('--seed', type=int)
    p.add_argument('--device', default='cuda', choices=['cpu', 'cuda'])
    p.add_argument('--resume')
    p.add_argument('--relation-weight', type=float)
    p.add_argument('--weight-form', choices=['full', 'gamma', 'uniform'])
    p.add_argument('--graph', choices=['shared', 'distance', 'nearest'])
    p.add_argument('--gradient-control', action='store_true')
    p.add_argument('--gradient-target', type=float)
    p.add_argument('--lmn', action='store_true')
    p.add_argument('--global-energy-file')
    a = p.parse_args()
    config = yaml.safe_load(Path(a.config).read_text())
    for name in ['mode', 'seed', 'relation_weight', 'weight_form', 'graph', 'gradient_target']:
        if getattr(a, name) is not None:
            config[name] = getattr(a, name)
    if a.gradient_control:
        config.update(gradient_control=True, mean_normalize=True, gam=False)
    if a.lmn:
        config.update(lmn=True, gam=False)
    if a.global_energy_file:
        config['global_energy'] = json.loads(Path(a.global_energy_file).read_text())['mean_energy']
    if config['mode'] == 'A11' and 'global_energy' not in config:
        p.error('A11 requires --global-energy-file from tools/calibrate_energy.py')
    if config.get('gradient_control') and config['mode'] not in ['M', 'degree', 'A2']:
        p.error('Gradient strength control requires a relation objective')
    from querydiff.training import run
    run(config, a.output, a.device, a.resume)


if __name__ == '__main__':
    main()
