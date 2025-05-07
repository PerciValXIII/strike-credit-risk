# STRIKE Experiment: Home Credit Default Risk

This directory applies STRIKE to the **Home Credit Default Risk** dataset from Kaggle. This is a large-scale, high-dimensional dataset (597 features) designed for default classification in a consumer finance context. The default rate is around 8%.

---

## Directory Layout

- This directory is split into:
  - `feature_engineering/`: Dataset-specific processing pipelines.
  - `feature_selection/`: Statistical and heuristic feature selection utilities.
  - `model_training/`: Code for training base and meta models.

---

## Goals

- Test STRIKE's scalability in high-dimensional, sparse, and noisy settings.
- Benchmark performance on a popular industry-grade competition dataset.

---

Performance metrics and AUC scores are detailed in Table 2 of the paper.
