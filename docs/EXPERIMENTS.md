# Training and evaluation

## Training variants

| Mode | Deep distillation / GAM | Query anchor | Local differences |
|---|---|---|---|
| `B` | Yes | No | No |
| `Q` | Yes | Yes | No |
| `M` | Yes | Yes | Energy/variation weighted |
| `A2` | Yes | Yes | Uniform weight |

## Configuration

The runner defaults to seed `20261011`, 128 training images, 32 development images, three epochs, FP32 detector training, AdamW with learning rate `1e-4` and weight decay `1e-4`, effective batch size four, and frozen batch-normalization statistics. Image membership comes from `data/research_protocol_v1/coco_split_A`; annotations come from the original ten-class VisDrone TXT files.

All variants use the same initialization, augmentation, and teacher brightness views. The detector input resolution is 640 × 640. Evaluation runs after each epoch; the final checkpoint is saved as `last.pth`.

Set `--epochs`, `--fit-limit`, `--dev-limit`, and `--seed` on the training command to configure a run. Image limit `0` selects the complete corresponding split. Use a distinct `--name` for each run.

## Evaluation conditions

| Condition | Image processing |
|---|---|
| `clean` | 640 × 640 input |
| `down2` | 640 → 320 → 640 |
| `down4` | 640 → 160 → 640 |
| `blur1` | Gaussian blur with radius 1 |
| `blur2` | Gaussian blur with radius 2 |

Evaluation uses the VisDrone Python toolkit with ignore-region handling, original-image coordinates, and up to 300 predictions per image. AP, AP50, AP75, AR1, AR10, AR100, and AR500 are reported in percentage units; all recall metrics use the available prediction set of up to 300 detections.

## Outputs

Each run writes to `results/<name>/`:

- `last.pth`: detector, projection, optimizer, and epoch state.
- `history.jsonl`: per-epoch metrics and mean training losses.
- `metrics_epoch_<epoch>_<condition>.json`: detection metrics.
- `predictions_epoch_<epoch>_<condition>.json`: image-level predictions.
- `provenance.json`: run arguments, image membership, and input hashes.
- `status.json`: training progress and execution status.
- `loss_scale_calibration.json`: fixed query-loss calibration values.
