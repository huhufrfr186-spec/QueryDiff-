# QueryDiff: Local Query Difference Distillation for Aerial Object Detection

QueryDiff transfers object-level semantic structure from a frozen vision teacher to an aerial object detector. It combines matched-query feature alignment with local query difference supervision, using teacher feature energy and brightness-view variation to weight local relations.

The detector is RT-DETRv4-S. CLIP ViT-B/16 and AnyUp construct teacher targets during training. Inference uses the trained detector directly, without the teacher or auxiliary projection heads.

## Method

1. Extract frozen CLIP features and upsample them with AnyUp.
2. Pool and normalize object features inside ground-truth boxes.
3. Associate final decoder queries with objects through Hungarian matching.
4. Align projected queries with object targets and supervise differences between nearby object pairs.
5. Combine query supervision with detection loss and deep feature distillation.

Equations and module descriptions are in [Method](docs/METHOD.md). Training variants and evaluation settings are in [Experiments](docs/EXPERIMENTS.md).

## Installation

Use Python 3.10 or later and install PyTorch and torchvision appropriate for your CUDA setup, then install the project dependencies:

```bash
python -m pip install -r requirements.txt
```

## Data and pretrained weights

Download the VisDrone2019-DET training data, the RT-DETRv4-S COCO checkpoint, CLIP ViT-B/16, and the AnyUp checkpoint from their respective sources listed in [Third-party sources](THIRD_PARTY_NOTICES.md). Arrange them as follows:

```text
QueryDiff/
├── data/
│   ├── VisDrone2019-DET-train/
│   │   ├── images/
│   │   └── annotations/
│   └── research_protocol_v1/coco_split_A/
│       ├── fit.json
│       └── development.json
├── models/
│   ├── rtdetrv4_s_official.pth
│   └── openai--clip-vit-base-patch16/
│       └── 57c216476eefef5ab752ec549e440a49ae4ae5f3/
│           ├── config.json
│           ├── preprocessor_config.json
│           └── pytorch_model.bin
└── research/2026_method_survey_20261010/upstream/anyup_archive/
    └── anyup_paper.pth
```

The split files define image membership; training reads the original ten-class VisDrone TXT annotations. Asset paths and SHA-256 checksums are listed in `assets_manifest.json`.

Check the data and model files before training:

```bash
python tools/check_assets.py
```

## Training

Run QueryDiff with energy/variation-weighted local differences:

```bash
python projects/rtdetrv4_uav/run_query_difference.py --mode M --name querydiff_M --epochs 3 --fit-limit 128 --dev-limit 32 --conditions
```

`--name` selects a new output directory under `results/`. Configure the epoch count and image limits for the intended experiment; a limit of `0` uses every image in the corresponding split. `--seed` controls the random seed and `--device` selects `cuda` or `cpu`.

| Mode | Deep feature distillation | Query anchor | Local query differences |
|---|---|---|---|
| `B` | Yes | No | No |
| `Q` | Yes | Yes | No |
| `M` | Yes | Yes | Energy/variation weighted |
| `A2` | Yes | Yes | Uniform weight |

Each epoch evaluates clean images. `--conditions` also evaluates reduced-resolution and blurred images after the final epoch. Run outputs include the detector checkpoint, training history, metrics, and predictions.

## Inference

```bash
python tools/infer_querydiff.py --checkpoint results/querydiff_M/last.pth --image path/to/image.jpg --output results/predictions.json --device cuda
```

The prediction JSON contains bounding boxes in original-image coordinates, confidence scores, and VisDrone category IDs from 1 to 10. The detector configuration and RT-DETRv4 initialization checkpoint should be placed at the paths above.

## Project structure

```text
code/                         Runtime paths and file utilities
projects/rtdetrv4_uav/         QueryDiff losses, teacher targets, training and analysis
tools/                        Asset checks and detector inference
docs/                         Method and experiment settings
data/research_protocol_v1/     Image split definitions
work/RT-DETRv4/                Detector implementation and configurations
work/visdrone-det-toolkit-python-api/  Detection evaluation
research/2026_method_survey_20261010/upstream/anyup_archive/  AnyUp
```

## Acknowledgments

QueryDiff uses RT-DETRv4, CLIP, AnyUp, and the VisDrone evaluation toolkit. Source links and license information are provided in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Third-party components retain their respective licenses.
