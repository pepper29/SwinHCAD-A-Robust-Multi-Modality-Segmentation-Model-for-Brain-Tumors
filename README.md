# SwinHCAD: A Robust Multi-Modality Segmentation Model for Brain Tumors Using Transformer and Channel-Wise Attention

[![Journal](https://img.shields.io/badge/Computers_Materials_and_Continua-Accepted-success)](https://www.sciencedirect.com/org/science/article/pii/S1546221825010987)
[![Paper](https://img.shields.io/badge/Paper-Link-blue)](https://www.sciencedirect.com/org/science/article/pii/S1546221825010987)

> **📢 News** > Our paper **"SwinHCAD: A Robust Multi-Modality Segmentation Model for Brain Tumors Using Transformer and Channel-Wise Attention"** has been accepted for publication in **Computers, Materials & Continua (CMC, 2025)**.

---

## 📌 Introduction

**SwinHCAD** is a state-of-the-art 3D segmentation framework designed to address the challenges of multi-modal brain tumor segmentation. It synergizes the long-range dependency modeling of **Swin Transformers** with a novel **Hierarchical Channel-Wise Attention Decoder (HCAD)**.

By dynamically recalibrating feature responses across different modalities (T1, T1ce, T2, FLAIR), SwinHCAD effectively highlights tumor regions while suppressing irrelevant background noise.

### 🏗️ Architecture
<p align="center">
  <img src="total framework.png" alt="SwinHCAD Architecture" width="95%"/>
  <br>
  <em>Figure 1: Overview of the SwinHCAD architecture.</em>
</p>

## ✨ Key Contributions

1.  **Hierarchical Channel-Wise Attention (HCAD):**
    * A novel decoding strategy that applies channel attention at multiple scales.
    * Effectively fuses semantic information from the decoder with high-resolution details from the encoder.
2.  **Robust Multi-Modality Learning:**
    * Automatically weighs the importance of each modality feature, making the model robust to data heterogeneity.
3.  **Superior Performance:**
    * Validated on **BraTS 2021** and **BraTS 2023** datasets, demonstrating superior segmentation accuracy compared to existing Transformer-based methods like SwinUNETR.

---

## 📂 Repository Structure

This repository provides the core implementation of the SwinHCAD architecture.

```bash
SwinHCAD/
├── models/
│   ├── swinhcad.py          # Main Network Assembly
│   └── hcad_block.py        # Hierarchical Channel Attention Module
├── requirements.txt         # Dependencies
└── README.md
```


## 📜 Citation
If you use this code or model in your research, please cite our paper:

```bash
@article{jin2025swinhcad,
  title={SwinHCAD: A Robust Multi-Modality Segmentation Model for Brain Tumors Using Transformer and Channel-Wise Attention},
  author={Jin, Seyong and Fayaz, Muhammad and Dang, L Minh and Song, Hyoung-Kyu and Moon, Hyeonjoon},
  journal={Computers, Materials and Continua},
  volume={86},
  number={1},
  pages={1--23},
  year={2025},
  publisher={Elsevier}
}

```

📝 License
This project is licensed under the MIT License.
