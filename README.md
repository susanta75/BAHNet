# BAHNet: Boundary-Aware Hybrid Network for Accurate Polyp Segmentation

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)

BAHNet pairs a SAM ViT-B semantic pathway with a lightweight convolutional pathway for boundary detail in colonoscopy images. Its main components are the Domain Adaptation Block (DAB) with Multi-Scale Spatial Enhancement (MSEB), Hierarchical Level Integration and Fusion (HLIF), Sub-Patch Preservation Encoder (SPPE), and Boundary-Aware Refinement Module (BARM) with Multi-scale Adaptive Edge Module (MAEM).

## Results and evaluation protocol

The paper reports the following results:

| Dataset | mDice | mIoU | F-measure | S-measure | E-measure | MAE |
|---|---:|---:|---:|---:|---:|---:|
| CVC-ClinicDB | 0.953 | 0.907 | 0.962 | 0.959 | 0.989 | 0.006 |
| Kvasir-SEG | 0.942 | 0.895 | 0.963 | 0.938 | 0.981 | 0.014 |
| CVC-ColonDB | 0.821 | 0.742 | 0.869 | 0.882 | 0.919 | 0.021 |
| ETIS-LaribPolypDB | 0.804 | 0.731 | 0.858 | 0.889 | 0.934 | 0.012 |

The training script selects checkpoints using the Kvasir, ClinicDB, and ETIS test splits. Scores on those datasets are therefore not independent held-out estimates. CVC-ColonDB is not used for checkpoint selection.

## Architecture

![BAHNet architecture](figures/model_architecture.jpg)

The primary model is `BAHNet` in `model/bahnet.py`. The encoder and decoder use SAM-derived components; local attribution and license terms are included in the source files and `LICENSE`.

## Setup

Use Python 3.10–3.12. Create a virtual environment and install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For CUDA, install the PyTorch build matching your system using the [official PyTorch selector](https://pytorch.org/get-started/locally/) before installing the remaining requirements.

## Dataset layout

The `Kvasir/train` folder contains the combined Kvasir and CVC-ClinicDB training images and masks. The separate ClinicDB test split is used for validation/checkpoint selection.

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
```

The dataset, SAM ViT-B checkpoint, trained checkpoints, and prediction outputs are external files; obtain them separately. Set the SAM checkpoint path in `train.py`.

## Training

```bash
python train.py
```

Check `Args.data_path`, `Args.encoder_ckpt`, and `Args.save_dir` in `train.py` first.

## Evaluation

```bash
python evaluation.py
```

Set `Args.checkpoint`, `Args.data_path`, and `Args.result_path` in `evaluation.py`. Predictions are written under `results/`, with metrics in `results/sod_metrics.csv`. Use a checkpoint trained with this BAHNet implementation.

## Citation

```bibtex
@article{bahnet2026,
  title   = {BAHNet: Boundary-Aware Hybrid Network for Accurate Polyp Segmentation in Colonoscopy Images},
  author  = {Khamrui, Susanta and Penhaker, Marek and Krejcar, Ondrej and Seal, Ayan},
  journal = {IEEE Journal of Biomedical and Health Informatics},
  year    = {2026},
  note    = {Under Review}
}
```

## License

This project is licensed under Apache-2.0; see `LICENSE`. The SAM-derived encoder components retain their upstream attribution.
