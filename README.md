# QueryDiff

**Local Query Difference Distillation for Dense Aerial Object Detection**

Yanting Hu, Qinyong Zeng, and Jiahao Wen  
School of Aeronautics and Astronautics, University of Electronic Science and Technology of China  
Corresponding author: Qinyong Zeng — qyzeng@uestc.edu.cn

QueryDiff transfers object-level relative structure from a frozen visual teacher to the final prediction queries of an aerial detector. Absolute object anchoring constrains individual fitting errors, while local vector differences couple neighboring query errors within teacher patches. Energy–repeatability weights allocate the difference supervision. Teacher networks and distillation projections are used during training; inference uses the detector alone.

## Method

The RT-DETRv4-S student uses 300 ordinary queries. Frozen CLIP ViT-B/16 patch features, extracted after `ln_post` and before the image–text projection, provide a 40×40×768 teacher map. AnyUp produces a 160×160 reference. Bilinear sampling at 4×4 bin centers forms an L2-normalized reference for each annotated object.

Final-layer Hungarian matching associates objects with prediction queries. A shared bias-free projection maps queries to the teacher feature space. Objects whose centers fall in the same 16-pixel teacher patch form an undirected local graph. The training objective is

```math
L = L_{det} + \lambda_e L_{deep} + \lambda_a L_{anchor} + \lambda_r L_{diff},
```

with `λa=1`, `λr=0.1`, and an initial deep coefficient of `20`. GAM updates the deep coefficient between epochs using the AIFI gradient share. [Method details](docs/METHOD.md) describe the loss reductions, weighting, and degree control.

## Installation

Use Python 3.11 or later with a compatible PyTorch/torchvision installation.

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Data and pretrained weights

Obtain [VisDrone2019-DET](https://github.com/VisDrone/VisDrone-Dataset), the [RT-DETRv4-S COCO checkpoint](https://github.com/RT-DETRs/RT-DETRv4), [OpenAI CLIP ViT-B/16](https://github.com/openai/CLIP), and [AnyUp](https://github.com/wimmerth/anyup/releases/download/checkpoint/anyup_paper.pth). Arrange the files as follows, or set their locations in `configs/visdrone.yaml`:

```text
QueryDiff/
├── data/
│   ├── VisDrone2019-DET-train/{images,annotations}/
│   ├── VisDrone2019-DET-val/{images,annotations}/
│   └── VisDrone2019-DET-test-dev/{images,annotations}/
└── weights/
    ├── rtdetrv4_s_coco.pth
    ├── ViT-B-16.pt
    └── anyup_paper.pth
```

CLIP uses the OpenAI `ViT-B-16.pt` checkpoint with SHA-256 `5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f`. Images preserve their aspect ratio and are centrally padded to 640×640. The training and validation loaders use their complete official splits and the original ten-class annotations.

## Training

Train on two GPUs with effective batch size 16:

```bash
torchrun --standalone --nproc_per_node=2 tools/train.py --config configs/visdrone.yaml --mode M --seed 0 --output runs/M_seed0
```

For one GPU, the same command can be launched with `python tools/train.py`; accumulation adjusts to preserve the effective batch size. The default schedule is 72 epochs, AdamW, learning rates `4e-4` and `2e-4` for main and backbone parameters, 1000 warmup updates, cosine decay to 1%, weight decay `1e-4`, and gradient clipping at `0.1`.

| Mode | Object anchor | Relation objective |
|---|---|---|
| `B` | Off | Off |
| `Q` | On | Off |
| `M` | On | Energy–repeatability-weighted vector differences |
| `A2` | On | Uniform vector differences |
| `degree` | On | Weighted-degree anchoring |
| `R` | Region anchor | Off |
| `C1` | Region anchor | Weighted differences |

```bash
python tools/train.py --mode Q --seed 0 --output runs/Q_seed0
python tools/train.py --mode degree --seed 0 --output runs/degree_seed0
python tools/train.py --mode M --weight-form gamma --relation-weight 1.6 --seed 3 --output runs/gamma_seed3
python tools/train.py --mode M --gradient-control --gradient-target 0.1 --seed 20 --output runs/normalized_M_seed20
```

`--gradient-control` uses per-image mean-normalized weights, fixes the deep coefficient at 20, and controls auxiliary gradient strength on deployed detector parameters. `--lmn` selects logarithmic minimum-norm combination. [Experiments](docs/EXPERIMENTS.md) lists the ablation controls and coefficient-selection settings.

Each run saves `last.pth`, `history.jsonl`, `config.json`, and `predictions.json`. Checkpoints include the detector, projections, optimizer, AMP scaler, and coefficient-controller state. Resume with `--resume runs/M_seed0/last.pth --output runs/M_seed0`.

## Prediction and evaluation

```bash
python tools/predict.py --checkpoint runs/M_seed0/last.pth --images data/VisDrone2019-DET-test-dev/images --output runs/M_seed0/test_predictions.json
python tools/evaluate.py --data data/VisDrone2019-DET-test-dev --predictions runs/M_seed0/test_predictions.json --protocol both --output runs/M_seed0/test_metrics.json
```

Prediction retains one highest-scoring class and one box per ordinary query, then stably ranks up to 300 valid predictions per image. Coordinates are mapped back to the original image; AP evaluation applies neither NMS nor a visualization score threshold.

`--protocol official` uses the VisDrone VOC precision-envelope integration and class-occurrence aggregation. `--protocol qde` uses QDE-AP300-v1, equal-class averaging at 101 recall points, and the defined subset-ignore order. Fixed geometric strata can be selected with `--subset small_shared`, `small_exclusive`, `larger_shared`, or `larger_exclusive`.

For degraded-image predictions, use `--condition down2`, `down4`, `blur1`, or `blur2` with `tools/predict.py`.

## Citation

```bibtex
@misc{hu2026querydiff,
  title={Local Query Difference Distillation for Dense Aerial Object Detection},
  author={Hu, Yanting and Zeng, Qinyong and Wen, Jiahao},
  year={2026},
  howpublished={\url{https://github.com/huhufrfr186-spec/QueryDiff-}}
}
```

## Acknowledgments

QueryDiff uses RT-DETRv4, CLIP, AnyUp, and the VisDrone evaluation toolkit. Attribution and license information are provided in [Third-party notices](THIRD_PARTY_NOTICES.md).
