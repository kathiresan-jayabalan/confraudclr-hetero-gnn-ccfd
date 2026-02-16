[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18618197.svg)](https://doi.org/10.5281/zenodo.18618197)

# ConFraudCLR-HeteroGNN-CCFD
Self-Supervised Contrastive Learning on Heterogeneous Graphs for Credit Card Fraud Detection

This repository provides a **Jupyter Notebook** implementation of a contrastive learning–based heterogeneous Graph Neural Network (GNN) pipeline for  **credit card fraud detection and evaluation** using the public Kaggle `creditcard.csv` dataset.

Main artifact:
confraudclr-hetero-gnn-ccfd.ipynb

End-to-end execution: data loading → heterogeneous graph construction → contrastive pretraining → supervised fine-tuning → evaluation

---

## Overview

Credit card fraud detection presents a challenging highly imbalanced classification problem where **learning robust representations from unlabeled transactions** can significantly improve detection performance.

This notebook implements a **two-stage learning framework** (ConFraudCLR) workflow that combines:

### Stage 1: Self-Supervised Contrastive Pretraining
- Constructs a **heterogeneous graph** with multiple node types:
  - `card` (cardholder)
  - `merchant`
  - `device`
- Defines heterogeneous relations:
  - `card → merchant`
  - `card → device`
- Applies **NT-Xent contrastive loss** (Normalized Temperature-scaled Cross-Entropy)
- Creates **augmented views** through:
  - Feature masking (random feature dropout)
  - Gaussian noise injection
  - Edge dropout in graph topology
- Learns discriminative node embeddings **without using fraud labels**

### Stage 2: Supervised Fine-Tuning
- Loads pretrained encoder weights
- Adds a **classification head** for binary fraud prediction
- Fine-tunes on labeled fraud data (with optional encoder unfreezing)
- Optionally applies **BorderlineSMOTE** oversampling to handle class imbalance
- Forms transaction representations by concatenating:
  - Card embedding
  - Merchant embedding
  - Device embedding
  - Transaction feature vector

All stages are implemented in a single notebook for clarity and reproducibility.

---

## Methodology

### Contrastive Pretraining Phase

1. **Data Augmentation**
   - Edge dropout: Randomly removes graph edges (preserves topology variability)
   - Feature masking: Randomly masks transaction features with probability `FEATMASKP`
   - Gaussian noise: Adds small noise (std = `GAUSSIANNOISESTD`) to node features

2. **Graph Construction**
   - Generate synthetic entity identifiers via hash-based bucketing from:
     - `Time` column
     - Binned `Amount` values
     - Selected PCA features (V1–V4)
   - Build heterogeneous graph with `PyTorch Geometric HeteroData`

3. **Contrastive Learning**
   - Forward two augmented views through heterogeneous GNN encoder (`HeteroConv` + `SAGEConv`)
   - Project embeddings through MLP projection head
   - Compute NT-Xent loss to maximize agreement between positive pairs
   - Save pretrained encoder: `confraudclrpretrained.pth`

### Supervised Fine-Tuning Phase

1. **Load and Preprocess**
   - Load pretrained encoder weights
   - Initialize classification head (2-layer MLP)
   - Optional: Apply BorderlineSMOTE to training set

2. **Training**
   - Freeze or unfreeze encoder weights (controlled by `FINETUNEUNFREEZEENCODER`)
   - Train with cross-entropy loss on fraud labels
   - Track validation metrics (ROC AUC, PR AUC, F1-score)
   - Save best model: `confraudclrbestfinetuned.pth`

3. **Evaluation**
   - Compute comprehensive metrics on held-out test set
   - Generate confusion matrix and classification report
   - Save final artifacts: `confraudclrfinal.pth`, `confraudclrartifacts.pkl`

---

## Dataset

The experiments use the public **Credit Card Fraud Detection** dataset (European cardholders, anonymized PCA features) available on Kaggle:

[https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)  
(subject to Kaggle terms of use)

After downloading, place the file at:

data/creditcard.csv

If the full dataset is unavailable, a small synthetic or reduced subset may be used to verify execution. Such data will not reproduce reported performance metrics.

---

## Architecture

┌─────────────────────────────────────────────┐
│         Contrastive Pretraining             │
│  ┌────────────┐        ┌────────────┐       │
│  │  View 1    │        │  View 2    │       │
│  │ (augmented)│        │ (augmented)│       │
│  └──────┬─────┘        └─────┬──────┘       │
│         │                    │              │
│  ┌──────▼────────────────────▼──────┐       │
│  │   Heterogeneous GNN Encoder      │       │
│  │   (HeteroConv + SAGEConv)        │       │
│  └──────┬────────────────────┬──────┘       │
│         │                    │              │
│  ┌──────▼──────┐      ┌──────▼──────┐       │
│  │ Projection  │      │ Projection  │       │
│  │    Head     │      │    Head     │       │
│  └──────┬──────┘      └──────┬──────┘       │
│         └────────┬────────────┘             │
│                  │                          │
│           NT-Xent Loss                      │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│        Supervised Fine-Tuning               │
│  ┌──────────────────────────────────┐       │
│  │  Pretrained Encoder (frozen or   │       │
│  │  unfrozen based on config)       │       │
│  └───────────────┬──────────────────┘       │
│                  │                          │
│  ┌───────────────▼──────────────────┐       │
│  │ Concatenate: Card + Merchant +   │       │
│  │ Device Embeddings + TX Features  │       │
│  └───────────────┬──────────────────┘       │
│                  │                          │
│  ┌───────────────▼──────────────────┐       │
│  │    Classification Head (MLP)     │       │
│  └───────────────┬──────────────────┘       │
│                  │                          │
│         Cross-Entropy Loss                  │
└─────────────────────────────────────────────┘

---

## Environment

**Tested configuration:**
- Python 3.8+
- Jupyter Notebook or Google Colab
- PyTorch 2.0+
- PyTorch Geometric 2.3+
- GPU: NVIDIA T4 GPU (cloud or Colab)

---

## Dependencies

Install dependencies from `requirements.txt`:
pip install -r requirements.txt

**Core packages:**
- torch>=2.0.0
- torch-geometric>=2.3.0
- torch-scatter>=2.1.0
- torch-sparse>=0.6.17
- torch-cluster>=1.6.1
- torch-spline-conv>=1.2.2
- numpy>=1.24.0
- pandas>=1.5.0
- scikit-learn>=1.2.0
- imbalanced-learn>=0.11.0 
- matplotlib>=3.7.0
- seaborn>=0.12.0
- tqdm>=4.65.0

---

## How to Run

1. Download the dataset from Kaggle.  
2. Place `creditcard.csv` under the `data/` directory.  
3. Launch Jupyter Notebook or JupyterLab.  
4. Open `notebooks/confraudclr-hetero-gnn-ccfd.ipynb`.  
5. Run all cells from top to bottom.

The notebook will preprocess data, construct the heterogeneous graph, train the GNN encoder and classifier, and evaluate the model.

Runtime depends on hardware and dataset size.

---

## Key Hyperparameters

**Configurable in the notebook:**
# Contrastive pretraining
PRETRAINEPOCHS = 100
PRETRAINBATCHSIZE = 2048
CONTRASTIVELR = 1e-3
TEMPERATURE = 0.5
EDGEDROPP = 0.1
GAUSSIANNOISESTD = 0.01
FEATMASKP = 0.1

# Supervised fine-tuning
FINETUNEEPOCHS = 100
FINETUNEBATCHSIZE = 2048
FINETUNELR = 1e-3
FINETUNEUNFREEZEENCODER = True  # Set False to freeze encoder

# Model architecture
EMBEDDIM = 64
NUMGNNLAYERS = 1
PROJDIM = 64

---

## Outputs

All evaluation metrics and visualizations are generated directly within the notebook output cells.

**Saved Artifacts:**
- `confraudclrpretrained.pth` – Pretrained encoder and projection head weights
- `confraudclrbestfinetuned.pth` – Best fine-tuned model (based on validation F1-score)
- `confraudclrfinal.pth` – Final model after all epochs
- `confraudclrartifacts.pkl` – Scaler, entity ID mappings, feature names, training history

**Evaluation Metrics:**

The notebook displays:
- ROC AUC (Receiver Operating Characteristic Area Under Curve)
- PR AUC (Precision-Recall Area Under Curve)
- Precision, Recall, F1-score
- Accuracy
- Confusion Matrix
- Per-class Classification Report

**Example performance (varies by configuration):**
- Test ROC AUC: ~0.94
- Test PR AUC: ~0.55
- Contrastive pretraining enables better feature representations
- Fine-tuning achieves high recall on minority fraud class

---

## Key Innovations

- Self-supervised pretraining on transaction graphs without labels
- NT-Xent contrastive loss with multi-view augmentations
- Heterogeneous graph structure capturing card-merchant-device relationships
- Two-stage learning (pretrain → fine-tune) for improved generalization
- Feature + topology augmentation for robust representations

---

## License

This project is licensed under the Apache License 2.0.

---

## Citation

If you use this repository in academic or research work, please cite:

Jayabalan, K. (2026). GraphFEN-CCFD: Heterogeneous GNN Model for Credit Card Fraud Detection (v1.0.1). Zenodo. https://doi.org/10.5281/zenodo.18618197
