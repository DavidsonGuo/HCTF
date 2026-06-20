# SpectraX-Inspect

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.12](https://img.shields.io/badge/TensorFlow-2.12-FF6F00.svg)](https://tensorflow.org/)

**SpectraX-Inspect** is an end-to-end full-stack hyperspectral expert system for automated, non-destructive evaluation of fungal contamination (e.g., *Aspergillus flavus*) in postharvest nuts. 

This repository contains the official implementation of the **HCTF (Hierarchical Cross-Transformer Fusion)** framework, seamlessly bridging deep learning algorithms, biochemical interpretability, and industrial SaaS deployment.

## 🌟 Key Features

- **Dual-Modal Feature Extraction:** Integrates PCA-guided spatial-spectral patches (PATCH) with Multi-dimensional Gramian Angular Fields (MGAF) to capture both local statistics and global non-linear spectral correlations.
- **HCTF Architecture:** Implements a Hierarchical Cross-Transformer Fusion mechanism using MGAF-Guided Self-Attention (MGT) and Cross-Modal Transformers (CMT).
- **Explainability-by-Architecture:** The deployment engine natively extracts physical attention weights during the forward pass, mapping fungal lipolytic activity directly onto the 2159–2216 nm biomarker window.
- **High-Throughput SaaS Deployment:** Powered by a FastAPI backend with in-memory asynchronous indexing and a clean frontend dashboard, supporting real-time macro population analytics and micro single-sample pathological traceability.

## 📂 Repository Structure

```text
SpectraX-Inspect/
├── demo_data/                          # 100-sample toy dataset for quick testing
│   ├── github_sample_PATCH.h5
│   ├── github_sample_MGAF.h5
│   └── github_sample_y.h5
├── App.py                              # FastAPI backend server & TIF figure exporter
├── Feature_extraction.py               # Preprocessing pipeline (ROI segmentation, PATCH, MGAF)
├── Model_def.py                        # HCTF model topology (Training & Explainable variants)
├── Train.py                            # Model training and evaluation pipeline
├── index.html                          # SaaS frontend dashboard (Open in any browser)
├── requirements.txt                    # Project dependencies
└── README.md                           # Project documentation
