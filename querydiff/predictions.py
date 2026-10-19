"""One class and one continuous box per ordinary query; no NMS."""
import torch


@torch.no_grad()
def decode(outputs, targets, maximum=300):
    result = []
    for logits, boxes, meta in zip(outputs['pred_logits'], outputs['pred_boxes'], targets):
        scores, labels = logits.float().sigmoid().max(-1)
        xyxy = torch.cat([boxes[:, :2]-boxes[:, 2:]/2, boxes[:, :2]+boxes[:, 2:]/2], -1).float()*640
        sx, sy, px, py = [float(v) for v in meta['transform']]
        h, w = [int(v) for v in meta['orig_size']]
        xyxy[:, [0, 2]] = ((xyxy[:, [0, 2]]-px)/sx).clamp(0, w)
        xyxy[:, [1, 3]] = ((xyxy[:, [1, 3]]-py)/sy).clamp(0, h)
        valid = torch.isfinite(xyxy).all(-1) & torch.isfinite(scores)
        valid &= (xyxy[:, 2] > xyxy[:, 0]) & (xyxy[:, 3] > xyxy[:, 1])
        ids = torch.arange(len(scores), device=scores.device)[valid]
        ids = ids[torch.argsort(scores[ids], descending=True, stable=True)[:maximum]]
        xywh = xyxy[ids].clone()
        xywh[:, 2:] -= xywh[:, :2]
        det = torch.cat([xywh, scores[ids, None], labels[ids, None]+1], dim=-1)
        result.append({'file_name': meta['file_name'], 'detections': det.cpu().tolist(),
                       'query_ids': ids.cpu().tolist()})
    return result
