# BAHNet: Boundary-Aware Hybrid Network for Small Polyp Segmentation via Foundation Model Adaptation

[![Paper](https://img.shields.io/badge/Paper-IEEE%20TMI-blue)](https://github.com/susanta75/BAHNet)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Abstract

BAHNet is a parameter-efficient CNN-transformer framework that addresses two structural 
limitations of the Segment Anything Model (SAM) simultaneously: the irrecoverable loss 
of sub-patch boundary detail caused by 16×16 patch tokenization, and the domain 
misalignment between SAM's natural-image pre-training and the colonoscopy domain. 
BAHNet introduces five components: Domain Adaptation Block (DAB), Multi-Scale Spatial 
Enhancement Block (MSEB), Hierarchical Level Integration and Fusion block (HLIF), 
Sub-Patch Preservation Encoder (SPPE), Boundary-Aware Refinement Module (BARM), 
and a Composite Loss with Spatial Consistency Regularization (SCR).

## Results

| Dataset | mDice | mIoU | MAE |
|---------|-------|------|-----|
| CVC-ClinicDB | 0.953 | 0.907 | 0.005 |
| Kvasir-SEG | 0.942 | 0.895 | 0.014 |
| CVC-ColonDB | 0.821 | 0.742 | 0.021 |
| ETIS-LaribPolypDB | 0.804 | 0.731 | 0.012 |

## Architecture

![BAHNet Architecture](figures/architecture.pdf)

## Code

> **The full source code, pretrained weights, and training/evaluation scripts will be 
> released upon acceptance of the paper.**
> 
> If you have any questions in the meantime, please open an issue or contact us at 
> skhamrui2002@gmail.com

## Citation

If you find this work useful, please consider citing:

```bibtex
@article{bahnet2026,
  title   = {BAHNet: Boundary-Aware Hybrid Network for Small Polyp Segmentation 
             via Foundation Model Adaptation},
  author  = {Khamrui, Susanta and Penhaker, Marek and Krejcar, Ondrej and Seal, Ayan},
  journal = {IEEE Transactions on Medical Imaging},
  year    = {2026},
  note    = {Under Review}
}
```

## Datasets

The following public datasets are used in this work:

- [Kvasir-SEG](https://datasets.simula.no/kvasir-seg/)
- [CVC-ClinicDB](https://polyp.grand-challenge.org/CVCClinicDB/)
- [CVC-ColonDB](http://vi.cvc.uab.es/colon-qa/cvccolondb/)
- [ETIS-LaribPolypDB](https://polyp.grand-challenge.org/EtisLarib/)

## Acknowledgements

The SAM backbone is based on the 
[Segment Anything Model](https://github.com/facebookresearch/segment-anything) 
by Meta AI Research.
