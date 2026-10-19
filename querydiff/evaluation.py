"""QDE-AP300-v1 and the native VisDrone VOC evaluation interface."""
from pathlib import Path
import sys
import numpy as np
from PIL import Image
from .data import read_annotations, valid_boxes


def overlaps(box, boxes, ioa=False):
    if not len(boxes):
        return np.empty(0)
    lo = np.maximum(box[:2], boxes[:, :2])
    hi = np.minimum(box[2:], boxes[:, 2:])
    intersection = np.maximum(0, hi-lo).prod(-1)
    area = np.maximum(0, box[2:]-box[:2]).prod()
    if ioa:
        return intersection/max(area, 1e-12)
    return intersection/np.maximum(area+np.maximum(0, boxes[:, 2:]-boxes[:, :2]).prod(-1)-intersection, 1e-12)


def geometric_partition(raw, width, height):
    boxes, valid = valid_boxes(raw, width, height)
    ids = np.flatnonzero(valid)
    original = boxes[valid].copy()
    b = original.copy()
    r = min(640/width, 640/height)
    nw, nh = round(r*width), round(r*height)
    b[:, [0, 2]] = b[:, [0, 2]]*(nw/width)+(640-nw)//2
    b[:, [1, 3]] = b[:, [1, 3]]*(nh/height)+(640-nh)//2
    px, py = (640-nw)//2, (640-nh)//2
    # Compare integer-coordinate products, so exact patch/size boundaries are
    # independent of floating-point cancellation after coordinate transforms.
    centers = (original[:, :2]+original[:, 2:])/2*np.array([nw/width, nh/height])+[px, py]
    cells = np.column_stack([
        np.floor(((original[:, 0]+original[:, 2])*nw+2*px*width)/(32*width)),
        np.floor(((original[:, 1]+original[:, 3])*nh+2*py*height)/(32*height)),
    ]).astype(int)
    counts = {tuple(c): int(np.sum(np.all(cells == c, axis=1))) for c in cells}
    shared = np.asarray([counts[tuple(c)] > 1 for c in cells], dtype=bool)
    small = ((original[:, 2]-original[:, 0])*nw <= 16*width) & ((original[:, 3]-original[:, 1])*nh <= 16*height)
    if len(b) > 1:
        distances = np.linalg.norm(centers[:, None]-centers[None, :], axis=-1)
        np.fill_diagonal(distances, np.inf)
        area = (original[:, 2]-original[:, 0])*(nw/width)*(original[:, 3]-original[:, 1])*(nh/height)
        q = distances.min(1)/np.sqrt(area)
    else:
        q = np.full(len(b), np.inf)
    return {'gt_ids': ids, 'small': small, 'shared': shared, 'q': q,
            'small_shared': small & shared, 'small_exclusive': small & ~shared,
            'larger_shared': ~small & shared, 'larger_exclusive': ~small & ~shared}


def qde_evaluate(directory, predictions, subset=None):
    directory = Path(directory)
    by_name = {p['file_name']: p for p in predictions}
    entries = []
    total, subset_total = np.zeros(10, int), np.zeros(10, int)
    for image_id, path in enumerate(sorted((directory/'images').glob('*.jpg'))):
        width, height = Image.open(path).size
        raw = read_annotations(directory/'annotations'/path.with_suffix('.txt').name)
        boxes, valid = valid_boxes(raw, width, height)
        ids = np.flatnonzero(valid)
        selected = np.ones(len(ids), dtype=bool) if subset is None else geometric_partition(raw, width, height)[subset]
        labels = raw[valid, 5].astype(int)
        for cls in range(1, 11):
            total[cls-1] += np.sum(labels == cls)
            subset_total[cls-1] += np.sum((labels == cls) & selected)
        prediction = by_name.get(path.name, {'detections': [], 'query_ids': []})
        det = np.asarray(prediction['detections'], float).reshape(-1, 6)
        query_ids = np.asarray(prediction.get('query_ids', list(range(len(det)))), int)
        order = np.lexsort((query_ids, -det[:, 4]))[:300]
        det, query_ids = det[order], query_ids[order]
        det[:, 2:4] += det[:, :2]
        entries.append((image_id, boxes[valid], labels, selected, boxes[~valid], det, query_ids, ids))
    ap, ar = np.zeros((10, 10)), np.zeros((10, 10))
    tp_by_class = np.zeros((10, 10), int)
    for cls in range(1, 11):
        for ti, threshold in enumerate(np.linspace(.5, .95, 10)):
            records = []
            for image_id, boxes, labels, selected, ignored, det, query_ids, ids in entries:
                primary = np.flatnonzero((labels == cls) & selected)
                secondary = np.flatnonzero((labels == cls) & ~selected)
                used = set()
                for d, query_id in zip(det, query_ids):
                    if int(d[5]) != cls:
                        continue
                    matched = False
                    for candidates, is_positive in [(primary, True), (secondary, False)]:
                        eligible = np.asarray([i for i in candidates if i not in used], int)
                        iou = overlaps(d[:4], boxes[eligible])
                        acceptable = np.flatnonzero(iou >= threshold)
                        if len(acceptable):
                            # eligible preserves annotation-row order, so argmax also fixes ties.
                            best = eligible[acceptable[np.argmax(iou[acceptable])]]
                            used.add(int(best))
                            if is_positive:
                                records.append((d[4], image_id, int(query_id), 1))
                            matched = True
                            break
                    if not matched and not np.any(overlaps(d[:4], ignored, ioa=True) >= .5):
                        records.append((d[4], image_id, int(query_id), 0))
            records.sort(key=lambda r: (-r[0], r[1], r[2]))
            flags = np.asarray([r[3] for r in records])
            tp = np.cumsum(flags == 1)
            fp = np.cumsum(flags == 0)
            recall = tp/max(1, subset_total[cls-1])
            precision = tp/np.maximum(tp+fp, 1)
            envelope = np.maximum.accumulate(precision[::-1])[::-1]
            indices = np.searchsorted(recall, np.linspace(0, 1, 101), side='left')
            sampled = np.zeros(101)
            inside = indices < len(envelope)
            sampled[inside] = envelope[indices[inside]]
            ap[cls-1, ti] = 100*sampled.mean()
            ar[cls-1, ti] = 100*recall[-1] if len(recall) else 0
            tp_by_class[cls-1, ti] = int(tp[-1]) if len(tp) else 0
    present = subset_total > 0
    mean = lambda a: float(a[present].mean()) if present.any() else None
    return {'protocol': 'QDE-AP300-v1', 'subset': subset or 'all', 'AP300': mean(ap),
            'AP50': mean(ap[:, 0]), 'AP75': mean(ap[:, 5]), 'AR300': mean(ar),
            'R50': float(100*tp_by_class[:, 0].sum()/max(1, subset_total.sum())),
            'R75': float(100*tp_by_class[:, 5].sum()/max(1, subset_total.sum())),
            'positives': subset_total.tolist(), 'TP50': tp_by_class[:, 0].tolist(),
            'class_AP': [float(v.mean()) if p else None for v, p in zip(ap, present)],
            'all_valid_objects': int(total.sum()), 'evaluated_objects': int(subset_total.sum())}


def official_visdrone(directory, predictions):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'third_party/visdrone_eval'))
    from viseval.eval_det import eval_det
    root = Path(directory)
    by_name = {p['file_name']: p for p in predictions}
    gts, detections, hs, ws = [], [], [], []
    for path in sorted((root/'images').glob('*.jpg')):
        width, height = Image.open(path).size
        raw = read_annotations(root/'annotations'/path.with_suffix('.txt').name).astype(np.int64)
        item = by_name.get(path.name, {'detections': [], 'query_ids': []})
        det = np.asarray(item['detections'], float).reshape(-1, 6)
        ids = np.asarray(item.get('query_ids', list(range(len(det)))))
        det = det[np.lexsort((ids, -det[:, 4]))[:500]]
        gts.append(raw.copy()); detections.append(det.copy()); hs.append(height); ws.append(width)
    values = eval_det(gts, detections, hs, ws, per_class=True)
    names = ['AP500', 'AP50', 'AP75', 'AR1', 'AR10', 'AR100', 'AR500']
    return dict(protocol='VisDrone-VOC', **{k: float(v) for k, v in zip(names, values[:7])},
                class_AP=np.asarray(values[7]).tolist(), images=len(gts))
