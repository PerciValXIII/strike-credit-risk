# STRIKE: A Feature-Group-Aware Stacking Framework for Credit Default Prediction

This repository contains the official implementation of the STRIKE framework proposed in the NeurIPS 2024 submission, **"STRIKE: Stacking via Targeted Representations of Isolated Knowledge Extractors"**. STRIKE is a modular stacking methodology that isolates semantically coherent feature groups, trains specialized base models per group, and combines their predictions through a meta-learner.

We provide scripts and configuration to replicate all experimental results on the following datasets:
- Polish Company Bankruptcy Dataset (UCI)
- LendingClub Peer-to-Peer Loan Dataset (Kaggle)
- Home Credit Default Risk Dataset (Kaggle)

---

## Setup Instructions

### Clone and Install Dependencies

```bash
git clone https://github.com/PerciValXIII/strike-credit-risk.git
cd strike-credit-risk
conda create -n strike python=3.10 -y
conda activate strike
pip install -r requirements.txt
```

---

## Reproduce Experiments

Each dataset experiment is contained in its own directory under `src/`.

### Run STRIKE on Polish Dataset

```bash
cd src/exp1_polish
python run_polish_strike.py
```

### Run STRIKE on LendingClub Dataset

```bash
cd src/exp2_lendingclub
python run_lendingclub_strike.py
```

### Run STRIKE on HomeCredit Dataset

```bash
cd src/exp3_homecredit
python run_homecredit_strike.py
```

Each script will:
- Load the corresponding dataset from `data/`
- Apply feature group isolation
- Train baseline models with out-of-fold prediction
- Construct the meta-dataset
- Train the final logistic regression meta-learner
- Output logs, trained models, and predictions to `outputs/`

---

## 🔧 Directory Structure

```
├── data/                    # Raw and processed datasets
├── outputs/
│   ├── logs/               # Training logs
│   ├── models/             # Pickled base/meta models
│   └── predictions/        # Final test set predictions
├── src/
│   ├── exp1_polish/
│   ├── exp2_lendingclub/
│   ├── exp3_homecredit/
│   ├── model_training/     # Model training utilities
│   ├── feature_engineering/
│   └── feature_selection/
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 📊 Evaluation

The primary evaluation metric is **AUC-ROC**. For all experiments:
- A 70/30 train-test split is used
- 5-fold cross-validation is applied within the training set
- OOF predictions are used to construct the meta-dataset
- All results are reproducible and stored under `outputs/`

---

## 📃 Citing

If you find STRIKE useful in your work, please cite:

```bibtex
@misc{maiti2024strike,
  author       = {Swattik Maiti},
  title        = {STRIKE: Stacking via Targeted Representations of Isolated Knowledge Extractors},
  year         = {2025},
  url          = {https://github.com/PerciValXIII/strike-credit-risk},
}
```

---

## 📎 Reference Datasets

- [Polish Bankruptcy Data (UCI)](https://archive.ics.uci.edu/dataset/365/polish+companies+bankruptcy+data)
- [LendingClub Loan Data (Kaggle)](https://www.kaggle.com/datasets/wordsforthewise/lending-club)
- [Home Credit Default Risk (Kaggle)](https://www.kaggle.com/competitions/home-credit-default-risk)

---

## 🔒 License

This project is licensed under the MIT License.

---
```
