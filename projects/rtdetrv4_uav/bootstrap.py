"""Use the existing environment and F-drive-only runtime directories."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code'))
from research_runtime import sha256, write_json
sys.path.insert(0, str(ROOT / 'work/rtdetrv4_deps'))
sys.path.insert(0, str(ROOT / 'work/RT-DETRv4'))

def result_dir(name):
    if not name or Path(name).name != name or '/' in name or '\\' in name or name in ('.', '..'):
        raise ValueError('Use a plain, new experiment name')
    output = ROOT / 'results' / name
    output.mkdir(exist_ok=False)
    return output
