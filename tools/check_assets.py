from pathlib import Path
import hashlib,json,sys
root=Path(__file__).resolve().parents[1]
manifest=json.loads((root/'assets_manifest.json').read_text())
failed=[]
for asset in manifest['excluded_assets']:
    p=root/asset['path']
    if not p.exists():failed.append('Missing: '+asset['path']);continue
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    if h.hexdigest()!=asset['sha256']:failed.append('Hash mismatch: '+asset['path'])
teacher=root/manifest['excluded_assets'][1]['path']
for name in manifest['teacher_sidecars']:
    if not (teacher.parent/name).exists():failed.append('Missing teacher sidecar: '+name)
for split in ['fit','development']:
    rows=json.loads((root/'data/research_protocol_v1/coco_split_A'/f'{split}.json').read_text())['images']
    for row in rows:
        for folder,name in [('images',row['file_name']),('annotations',str(Path(row['file_name']).with_suffix('.txt')))]:
            if not (root/'data/VisDrone2019-DET-train'/folder/name).exists():failed.append(f'Missing dataset {folder}/{name}')
print(json.dumps({'ok':not failed,'problems':failed},ensure_ascii=False,indent=2))
sys.exit(bool(failed))
