from pathlib import Path
import argparse
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from querydiff.evaluation import qde_evaluate, official_visdrone
from querydiff.training import write_json


def main():
    p = argparse.ArgumentParser(description='Evaluate exported predictions on a complete VisDrone split.')
    p.add_argument('--data', required=True)
    p.add_argument('--predictions', required=True)
    p.add_argument('--protocol', choices=['qde', 'official', 'both'], default='both')
    p.add_argument('--subset', choices=['small_shared', 'small_exclusive', 'larger_shared', 'larger_exclusive'])
    p.add_argument('--output', required=True)
    a = p.parse_args(); predictions = json.loads(Path(a.predictions).read_text())
    result = {}
    if a.protocol in ['qde', 'both']:
        result['qde'] = qde_evaluate(a.data, predictions, subset=a.subset)
    if a.protocol in ['official', 'both']:
        result['official'] = official_visdrone(a.data, predictions)
    write_json(a.output, result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
