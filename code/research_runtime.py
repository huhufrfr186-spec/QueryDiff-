"""Project-local runtime; never install packages or write caches outside F:."""
import hashlib
import json
import os
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
for key, folder in {
    'YOLO_CONFIG_DIR': 'ultralytics', 'MPLCONFIGDIR': 'matplotlib',
    'TORCH_HOME': 'torch', 'HF_HOME': 'huggingface', 'TEMP': 'tmp', 'TMP': 'tmp',
    'XDG_CACHE_HOME': 'cache', 'TORCHINDUCTOR_CACHE_DIR': 'torchinductor',
    'TRITON_CACHE_DIR': 'triton', 'PYTHONPYCACHEPREFIX': 'pycache',
    'CUDA_CACHE_PATH': 'cuda_cache', 'TORCH_EXTENSIONS_DIR': 'torch_extensions',
}.items():
    target = ROOT / 'work' / folder
    target.mkdir(parents=True, exist_ok=True)
    os.environ[key] = str(target)
os.environ['YOLO_OFFLINE'] = 'true'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
for folder in ('work/deps', 'work/localpkgs'):
    sys.path.insert(0, str(ROOT / folder))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.' + uuid.uuid4().hex + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    for attempt in range(12):
        try:
            temp.replace(path)
            return
        except PermissionError:
            # Windows readers/antivirus can briefly hold the destination open.
            if attempt==11:
                raise
            time.sleep(min(.05*2**attempt,.5))
