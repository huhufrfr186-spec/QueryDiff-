# Experiment settings

The main VisDrone configuration uses all 6471 training images, 548 validation images, and ten classes. Test-dev evaluation uses all 1610 images. Final epoch-72 checkpoints are evaluated after validation-based configuration selection.

Training uses AdamW with main/backbone learning rates 4e-4/2e-4, weight decay 1e-4, betas (0.9,0.999), 1000 linear warmup updates, cosine decay to 1%, AMP, clipping at 0.1, microbatch two per worker, and effective batch size 16. Standard augmentation uses a horizontal flip probability of 0.5 and color jitter strength 0.2. The strong-augmentation option uses jitter 0.4 and blur probability 0.1 during the first 60 epochs.

## Main configurations

| Mode | Anchor | Relation |
|---|---|---|
| B | 0 | 0 |
| Q / A1 | 1 | 0 |
| M | 1 | 0.1, full weights |
| degree | 1 | 0.1, weighted-degree term |
| A2 | 1 | 0.1, uniform differences |
| A3 | 1 | All matched pairs |
| A4 | 1 | Same edge count sampled from all pairs |
| A5 | 1 | Weighted scalar edge-length error |
| A6 | 1 | Weighted edge-cosine error |
| A7 | 1 | Teacher differences shuffled within image |
| A8 | 0 | 0.1, full differences |
| A9 | 1 | Bilinear 160×160 references |
| A10 | 1 | Zero photometric variation in weight |
| A11 | 1 | Fixed mean teacher energy in weight |
| R | Region anchor | 0 |
| C1 | Region anchor | Full differences |

A11 uses the mean nonempty per-image edge energy from the first 100 training batches in fixed seed-zero image order. Compute it once and share the resulting constant:

```bash
python tools/calibrate_energy.py --config configs/visdrone.yaml --output runs/mean_energy.json
python tools/train.py --mode A11 --global-energy-file runs/mean_energy.json --output runs/A11_seed0
```

The principal full/gamma weight search uses factors `[0.25,0.5,1,2,4]`, centers 0.1/1.6, and selection seeds 3 and 7. Confirmation uses paired seeds 0,1,2. Fixed-gradient confirmation uses mean-normalized weights, a fixed deep coefficient 20, auxiliary gradient target 0.1, and paired seeds 20–24. `--graph distance` matches the shared graph's 2-pixel center-distance-bin counts; `--graph nearest` selects the same number of nearest pairs.

## Metrics

Official VisDrone evaluation uses the native ignored-region routine, VOC precision-envelope integration, imagewise maxDets 500, and class-occurrence aggregation. The detector supplies at most 300 predictions.

QDE-AP300-v1 uses one class per query, imagewise top-300 predictions, equal-class averaging at 101 recall points and ten IoU thresholds from 0.50 through 0.95. A class without positives is excluded. Subset evaluation matches subset positives first, then ignores one-to-one same-class objects outside the subset, then applies native ignored-region overlap. R50/R75 are object-microaveraged recalls over fixed annotation populations.

The small/shared partition is constructed before prediction matching. Small objects have both letterboxed width and height at most 16 pixels. Shared objects have another valid center in the same teacher patch. Crowding uses nearest-center distance divided by the square root of box area, independently of patch membership.

## Numerical checks

```bash
python -m pip install pytest
python -m pytest tests -q
```

The tests cover the graph quadratic and gradient, equal diagonal Hessians of difference/degree objectives, zero-energy and empty-edge behavior, region sampling, component decomposition, GAM, LMN, one-class decoding, letterbox inversion, ignored regions, recall integration, and class aggregation.
