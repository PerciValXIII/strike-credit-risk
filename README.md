# STRIKE: A Feature-Group-Aware Stacking Framework for Credit Default Prediction

This repository implements **STRIKE** (Stacking via Targeted Representations of Isolated Knowledge Extractors), a modular and scalable machine learning framework designed for robust credit risk default prediction. STRIKE introduces a novel stacking architecture that partitions the feature space into semantically meaningful groups (e.g., demographic, credit history) before training specialized base learners, which are then fused using a meta-model for final predictions.

The pipeline has been evaluated across three real-world datasets—Polish bankruptcy data, LendingClub P2P loans, and the Home Credit Default Risk dataset—demonstrating consistent improvements over classical and deep learning baselines.

---


## Datasets

- **Polish** (UCI): Company bankruptcy prediction using financial ratios.
- **LendingClub** (Kaggle): P2P loan performance with moderate imbalance.
- **HomeCredit** (Kaggle): Consumer loan data with high sparsity and imbalance.

---

## How to Run

1. Clone the repository and navigate to the root directory.
2. Ensure all dependencies (Python 3.10+, scikit-learn, XGBoost, LightGBM) are installed.
3. Use the experiment notebooks inside `src/exp*/` for dataset-specific execution.

---