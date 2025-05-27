
# STRIKE Experiment: Home Credit Default Risk

This module applies **STRIKE**, a multi-model stacking framework, to the **Home Credit Default Risk** dataset from Kaggle. The dataset contains over 300,000 loan applications and hundreds of engineered features designed for binary classification of consumer loan default. The default rate is approximately 8%.

---

## Objectives

- Validate STRIKE's performance on a high-dimensional, sparse, and noisy real-world credit scoring dataset.
- Test the pipeline's robustness under class imbalance and many redundant features.
- Reproduce model-specific metrics discussed in Section 4.5 of the NeurIPS paper.

---

## Running the Experiment

### Step 0: Prepare Raw Data

Before running any code, download the raw CSV files from the following Google Drive link:

```
https://drive.google.com/drive/folders/1ef8jGMQSni6r4pSW2aMg6c453Y2a5tt6?usp=sharing
```

Then, **create a `data/` folder inside `exp3_homecredit/` directory** and place all the downloaded raw files from `exp3_homecredit_data` inside the data folder. 

---

### Step 1: Feature Engineering

Run the following two scripts to generate intermediate features:

```bash
python src/exp3_homecredit/feature_engineering/sm_deq_feature_creation.py
python src/exp3_homecredit/feature_engineering/sm_vin_feature_creation.py
```

These will produce:

- `data/deq_features_level1.csv`
- `data/vintage_features_1.csv`

---

### Step 2: Train Base & Meta Models

Run the main stacking pipeline to preprocess features and train all models:

```bash
python src/exp3_homecredit/model_training/model_stacking_run.py
```

This will:

- Preprocess the engineered datasets
- Train base models on AMTF, DEQ, and Vintage features
- Train a meta-model on stacked predictions
- Save logs to `logs/strike_homecredit_stacking.log`

---

## Notes

- Ensure all raw CSVs (e.g., `application_train.csv`, `bureau.csv`, etc.) are inside the `data/` directory.
- All intermediate and processed feature files will also be saved to the same `data/` folder.
- Logs are saved to the `logs/` folder inside `exp3_homecredit`.

---

For performance metrics and AUC results, refer to Table 2 in the NeurIPS paper.
