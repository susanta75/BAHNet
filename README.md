# BAHNet: Boundary-Aware Hybrid Network

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)

BAHNet is a hybrid network for polyp segmentation in colonoscopy images. It combines a SAM ViT-B image encoder with a parallel convolutional path for fine spatial details. The implementation includes a Domain Adaptation Block (DAB) with Multi-Scale Spatial Enhancement (MSEB), Hierarchical Level Integration and Fusion (HLIF), Sub-Patch Preservation Encoder (SPPE), and Boundary-Aware Refinement Module (BARM) with Multi-scale Adaptive Edge Module (MAEM).

## Architecture

![BAHNet architecture](figures/model_architecture.jpg)

The model is implemented in `model/bahnet.py`; its network blocks are in `model/block.py`. The training loss combines the segmentation loss with side-output supervision and PSL regularization (`utils/loss.py`).

## Requirements and setup

Use Python 3.10–3.12. Create and activate a virtual environment, then install the listed dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install a PyTorch build that matches your CPU or CUDA setup using the [official PyTorch instructions](https://pytorch.org/get-started/locally/), then install the project requirements:

```bash
pip install -r requirements.txt
```

## Data and pretrained weights

The scripts expect images and masks in matching `image/` and `mask/` directories. The training loader reads `train/`; validation and evaluation loaders read `test/`. By default, `dataset/SOD/Kvasir/train/` contains the combined Kvasir and CVC-ClinicDB training images and masks, as arranged for this experiment. Keep the combined training set in this `Kvasir` folder; the scripts do not load a second ClinicDB training folder.

Expected layout:

```text
dataset/SOD/
  Kvasir/
    train/image/
    train/mask/
    test/image/
    test/mask/
  CVC-ClinicDB/
    test/image/
    test/mask/
  CVC-ColonDB/
    test/image/
    test/mask/
  ETIS-Lari/
    test/image/
    test/mask/
pretrained/
  sam_vit_b_01ec64.pth
```

Obtain the datasets and SAM ViT-B checkpoint separately if they are not already available locally. The training script uses `./pretrained/sam_vit_b_01ec64.pth` by default. Dataset and checkpoint paths can be changed in the `Args` classes in `train.py` and `evaluation.py`.

## Training

Set the paths in `train.py`, then run:

```bash
python train.py
```

The default configuration uses 512-pixel inputs, batch size 16, 100 epochs, and AdamW. Checkpoints and a CSV training log are written to `output/`. Training uses `Kvasir/train`; checkpoint selection averages MAE over the Kvasir, ClinicDB, and ETIS test splits.

**Evaluation protocol note:** those three test splits are used during checkpoint selection, so their reported scores are not independent held-out results. CVC-ColonDB is evaluated by `evaluation.py` but is not part of the checkpoint criterion.

## Evaluation

Set `Args.checkpoint`, `Args.data_path`, and `Args.result_path` in `evaluation.py`, then run:

```bash
python evaluation.py
```

The script evaluates Kvasir, CVC-ClinicDB, CVC-ColonDB, and ETIS-Lari. It saves predicted masks under `results/<dataset>/` and appends metrics to `results/sod_metrics.csv`. Use a checkpoint produced by this BAHNet implementation.

## Reported paper results

| Dataset | mDice | mIoU | F-measure | S-measure | E-measure | MAE |
|---|---:|---:|---:|---:|---:|---:|
| CVC-ClinicDB | 0.953 | 0.907 | 0.962 | 0.959 | 0.989 | 0.006 |
| Kvasir-SEG | 0.942 | 0.895 | 0.963 | 0.938 | 0.981 | 0.014 |
| CVC-ColonDB | 0.821 | 0.742 | 0.869 | 0.882 | 0.919 | 0.021 |
| ETIS-LaribPolypDB | 0.804 | 0.731 | 0.858 | 0.889 | 0.934 | 0.012 |

These are the values reported in the paper; results can vary with data preparation, checkpoint, and evaluation settings. See the evaluation protocol note above when interpreting the ClinicDB, Kvasir, and ETIS scores.

## License

This project is licensed under Apache-2.0; see [LICENSE](LICENSE). SAM-derived components retain their upstream attribution in the source files.
