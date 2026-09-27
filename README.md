# Enhancing Origin–Destination Flow Prediction via Bi-Directional Spatio-Temporal Inference and Interconnected Feature Evolution

[![DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.eswa.2024.125679-blue)](https://doi.org/10.1016/j.eswa.2024.125679)
[![Paper page](https://img.shields.io/badge/paper-page-blue)](https://codezx6.github.io/papers/bist-if.html)

Official implementation of **BiST-IF** (Expert Systems with Applications 2025): *Enhancing origin–destination flow prediction via bi-directional spatio-temporal inference and interconnected feature evolution*.

BiST-IF predicts origin–destination (OD) flows between metro stations or urban areas by correcting delayed recent OD matrices, applying bi-directional origin/destination attention, and fusing arrival-side (Out-OD) flows through an attention-based mutual information mechanism. It lowers MAE by an average of 7.55% on HZMetro relative to the best baseline.

📄 Paper: https://doi.org/10.1016/j.eswa.2024.125679 · 🌐 Paper page with quoted results, FAQ and BibTeX: https://codezx6.github.io/papers/bist-if.html


A deep learning framework for origin–destination (OD) flow prediction that explicitly models flow delay, bi-directional spatio-temporal dependencies, and mutual information between OD and Out-OD flows.

---

## Overview

**BiST-IF** is a spatio-temporal OD flow prediction model for intelligent transportation systems.
Unlike conventional OD prediction methods that rely solely on departure-based OD matrices, BiST-IF jointly models **OD flow** and **Out-OD flow** to correct delayed data, capture periodic patterns, and supplement the arrival information missing from OD flow.

The model decomposes OD prediction into **periodic patterns (weekly, daily)** and **short-term fluctuations**, integrating them through bi-directional attention and interconnected feature evolution.

---

## Model Architecture

BiST-IF consists of three main modules:

1. **OD Delay Correction (ODDC)**
    - Estimates delayed OD distribution using historical periodic patterns and recent Out-OD flows
    - Completes recent OD matrices to reduce cumulative delay bias

2. **OD Bi-directional Attention (ODBA)**
    - BiLSTM-based OD flow trend extraction
    - A single attention map computed between origin and destination flow features
    - Spatio-temporal feature extraction with convolutional layers

3. **Mutual Information Flow Evolution (MFE)**
    - Out-OD feature extraction
    - Multi-head mutual information attention between OD and Out-OD
    - Gated feature update for temporal evolution

Final predictions are obtained by fusing outputs from weekly, daily, recent OD streams and the MFE module.

---

## Datasets

BiST-IF is evaluated on two real-world datasets:

- **HZMetro**
    - 80 metro stations (Hangzhou)
    - 10-minute intervals
    - January 2019

- **NYC-TOD2018**
    - 69 Manhattan taxi zones
    - 30-minute intervals
    - February–April 2018

---

## Citation

If you use this work, please cite:

```bibtex
@article{yu2025bistif,
  title        = {Enhancing origin–destination flow prediction via bi-directional spatio-temporal inference and interconnected feature evolution},
  author       = {Yu, Piao and Zhang, Xu and Gong, Yongshun and Zhang, Jian and Sun, Haoliang and Zhang, Junjie and Zhang, Xinxin and Yin, Yilong},
  journal      = {Expert Systems with Applications},
  year         = {2025},
  volume       = {264},
  pages        = {125679},
  doi          = {10.1016/j.eswa.2024.125679},
  issn         = {0957-4174},
  url          = {https://doi.org/10.1016/j.eswa.2024.125679}
}
```

---

## License

This project is licensed under the MIT License.

---

## Contact

For questions or discussions, please open an issue in the repository.
