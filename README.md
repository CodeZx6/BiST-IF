# Enhancing Origin–Destination Flow Prediction via Bi-Directional Spatio-Temporal Inference and Interconnected Feature Evolution

A deep learning framework for origin–destination (OD) flow prediction that explicitly models flow delay, bi-directional spatio-temporal dependencies, and mutual information between OD and Out-OD flows.

---

## Overview

**BiST-IF** is a spatio-temporal OD flow prediction model for intelligent transportation systems.
Unlike conventional OD prediction methods that rely solely on departure-based OD matrices, BiST-IF jointly models **OD flow** and **Out-OD flow** to correct delayed data, capture periodic patterns, and enhance destination-aware inference.

The model decomposes OD prediction into **periodic patterns (weekly, daily)** and **short-term fluctuations**, integrating them through bi-directional attention and interconnected feature evolution.

---

## Model Architecture

BiST-IF consists of three main modules:

1. **OD Delay Correction (ODDC)**
    - Estimates delayed OD distribution using historical periodic patterns and recent Out-OD flows
    - Completes recent OD matrices to reduce cumulative delay bias

2. **OD Bi-directional Attention (ODBA)**
    - BiLSTM-based OD flow trend extraction
    - Origin-wise and destination-wise attention computation
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
@article{yu2025enhancing,
  title={Enhancing origin--destination flow prediction via bi-directional spatio-temporal inference and interconnected feature evolution},
  author={Yu, Piao and Zhang, Xu and Gong, Yongshun and Zhang, Jian and Sun, Haoliang and Zhang, Junjie and Zhang, Xinxin and Yin, Yilong},
  journal={Expert Systems with Applications},
  volume={264},
  pages={125679},
  year={2025},
  publisher={Elsevier}
}
```

---

## License

This project is licensed under the MIT License.

---

## Contact

For questions or discussions, please open an issue in the repository.
