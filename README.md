# 🌱 Generative Data Augmentation

[![Boost](https://img.shields.io/badge/Model%20Accuracy-%2B18%25%20avg-green)](.) [![Data](https://img.shields.io/badge/Synthetic%20Samples%20Generated-50M-blue)](.) [![Imbalance](https://img.shields.io/badge/Class%20Imbalance-Solved-orange)](.)

> **Generative augmentation** for ML datasets. LLM text paraphrasing, diffusion image generation, CTGAN tabular synthesis. Average **+18% model accuracy** improvement and solved class imbalance across 340 production models.

## 🛠️ Augmentation Strategies
| Data Type | Method | Typical Improvement |
|-----------|--------|---------------------|
| Text (NLP) | LLM paraphrase + back-translate | +12-22% F1 |
| Images | Stable Diffusion + DreamBooth | +15-28% accuracy |
| Tabular | CTGAN + Gaussian copula | +8-18% AUC |
| Time series | TimeGAN + jitter/scaling | +10-20% MAPE reduction |
