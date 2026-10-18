"""Reuse ten-class predictions for grid occupancy and object-weighted recall."""
from bootstrap import ROOT, write_json, sha256
from pathlib import Path
import argparse, json, sys
from collections import Counter
import numpy as np

TOOL=next((ROOT/'work/visdrone-det-toolkit-python-api').glob('*/viseval'))
sys.path.insert(0,str(TOOL.parent))
np.int=int
from viseval.drop_objects_in_igr import drop_objects_in_igr
from viseval.calc_accuracy import eval_res


def analyze(run, condition='clean'):
    provenance=json.loads((run/'provenance.json').read_text())
    if provenance['arguments']['dev_limit']!=32:
        raise ValueError('This analysis is registered for the current 32-image pilot')
    epoch=provenance['arguments']['epochs']
    predictions=run/f'predictions_epoch_{epoch:03d}_{condition}.json'
    rows=json.loads(predictions.read_text())
    assert {r['file_name'] for r in rows}==set(provenance['development'])
    membership=ROOT/'data/research_protocol_v1/coco_split_A/development.json'
    images={r['file_name']:r for r in json.loads(membership.read_text())['images']}
    totals=Counter();hits={.5:Counter(),.75:Counter()};matches={.5:Counter(),.75:Counter()}
    image_stats=[];annotation_hashes={}
    for row in rows:
        meta=images[row['file_name']]
        annotation=ROOT/'data/VisDrone2019-DET-train/annotations'/Path(row['file_name']).with_suffix('.txt')
        annotation_hashes[row['file_name']]=sha256(annotation)
        gt=np.array([[int(v) for v in line.split(',')[:8]] for line in annotation.read_text().splitlines() if line.strip()],dtype=np.int32).reshape(-1,8)
        det=np.asarray(row['detections'],dtype=np.float32).reshape(-1,6)
        assert len(det)<=300
        det=det[np.argsort(-det[:,4])]
        gt,det=drop_objects_in_igr(gt,det,meta['height'],meta['width'])
        gt=gt.copy();det=det.copy()
        valid=(gt[:,4]>0)&(gt[:,5]>=1)&(gt[:,5]<=10)&(gt[:,2]>0)&(gt[:,3]>0)
        center=(gt[:,:2]+gt[:,2:4]/2)/np.array([meta['width'],meta['height']])
        shared={}
        for grid in (40,80):
            cell=np.clip(np.floor(center*grid).astype(np.int64),0,grid-1)
            counts=Counter(map(tuple,cell[valid]))
            shared[grid]=np.array([counts[tuple(c)]>1 for c in cell])
        assert not np.any(shared[80][valid]&~shared[40][valid]),'Aligned grids must be nested'
        small=(gt[:,2]/meta['width']*640<=16)&(gt[:,3]/meta['height']*640<=16)
        strata=[('subpatch' if small[i] else 'larger')+'_'+('shared' if shared[40][i] else 'single') for i in range(len(gt))]
        totals.update(strata[i] for i in np.flatnonzero(valid))
        image_stats.append(dict(file_name=row['file_name'],valid_objects=int(valid.sum()),
            teacher_shared_objects=int((shared[40]&valid).sum()),student_shared_objects=int((shared[80]&valid).sum()),
            objects_above_query_budget=max(0,int(valid.sum())-300)))
        for category in range(1,11):
            class_indices=np.flatnonzero(gt[:,5]==category)
            class_gt=gt[class_indices,:5].astype(np.float32,copy=True)
            class_gt[:,4]=1-class_gt[:,4]
            class_det=det[det[:,5]==category,:5].copy()
            normal_indices=class_indices[class_gt[:,4]==0]
            for threshold in (.5,.75):
                matched_gt,matched_det=eval_res(class_gt.copy(),class_det.copy(),threshold)
                # eval_res preserves order of regular GT and appends ignored GT.
                for source_index,match in zip(normal_indices,matched_gt[:len(normal_indices),4]):
                    if valid[source_index] and match==1:hits[threshold][strata[source_index]]+=1
                matches[threshold].update(dict(true_positive=int((matched_det[:,5]==1).sum()),
                    false_positive=int((matched_det[:,5]==0).sum()),ignored_prediction=int((matched_det[:,5]==-1).sum())))
    n=sum(totals.values());teacher_n=sum(r['teacher_shared_objects'] for r in image_stats)
    student_n=sum(r['student_shared_objects'] for r in image_stats)
    return dict(scope='Ten-class three-epoch single-seed development pilot; descriptive object recall, not AP or official AR500',
        matching='Official Python-port eval_res; ignore-region preprocessing; class-wise score-order one-to-one matching; all top300 predictions, no score threshold',
        primary_matching_source='https://github.com/VisDrone/VisDrone2018-DET-toolkit/blob/master/utils/calcAccuracy.m',
        image_count=len(rows),valid_objects=n,teacher_grid=[40,40],student_grid=[80,80],input_size=[640,640],
        teacher_shared_objects=teacher_n,teacher_shared_fraction=teacher_n/max(1,n),
        student_shared_objects=student_n,student_shared_fraction=student_n/max(1,n),
        aligned_grid_relation='The stride8 grid is nested in stride16; the OR edge set equals the teacher-cell edge set in this resize protocol',
        strata={key:dict(objects=value,recall_iou50=hits[.5][key]/value,recall_iou75=hits[.75][key]/value,
                         matched_iou50=hits[.5][key],matched_iou75=hits[.75][key]) for key,value in sorted(totals.items())},
        match_counts={str(k):dict(v) for k,v in matches.items()},images=image_stats,
        trace=dict(predictions_sha256=sha256(predictions),provenance_sha256=sha256(run/'provenance.json'),
                   original_annotations_sha256=annotation_hashes),
        interpretation='Occupancy and recall describe this checkpoint; they do not identify a causal grid effect or demonstrate a proposed-method gain')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);args=p.parse_args()
    run=(ROOT/'results'/args.run).resolve()
    if run.parent!=(ROOT/'results').resolve():raise ValueError('Use a project result directory name')
    result=analyze(run);write_json(run/'grid_object_recall_clean.json',result)
    print(json.dumps({key:result[key] for key in ('valid_objects','teacher_shared_fraction','student_shared_fraction','strata')},ensure_ascii=False))
