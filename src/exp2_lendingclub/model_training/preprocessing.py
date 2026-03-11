# src/exp2_lendingclub/model_training/preprocessing.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from .config import (
    DATA_PATH,
    PROCESSED_TRAIN,
    PROCESSED_TEST,
    GROUP_DATA_DIR,
    FEATURE_GROUPS,
    TARGET,
    ID_COL,
)
from .utils_logger import get_logger


def main():
    logger = get_logger("preprocessing", "preprocessing_lendingclub.log")
    logger.info("🚀 Starting preprocessing for LendingClub")

    # 1. Load raw data
    raw = pd.read_csv(DATA_PATH)
    raw.columns = raw.columns.str.strip()
    logger.info(f"Loaded raw data: {raw.shape}")

    # 2. Separate meta & features
    meta_cols = [c for c in [ID_COL, TARGET] if c in raw.columns]
    meta = raw[meta_cols].copy()
    df = raw.drop(columns=["emp_title"] + meta_cols, errors="ignore")
    logger.info(f"Feature matrix shape before split: {df.shape}")

    # 3. Stratified train/test split
    X_train, X_test, meta_train, meta_test = train_test_split(
        df,
        meta,
        test_size=0.2,
        random_state=42,
        stratify=meta[TARGET],
    )
    logger.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # 4. Drop high-NA columns ( > 90% NA ) based on TRAIN
    thresh = int(0.1 * len(X_train))  # need ≥10% non-null to keep
    high_na = X_train.columns[X_train.isna().sum() > (len(X_train) - thresh)].tolist()
    if high_na:
        logger.info(f"Dropping columns > 90%% NaN (train-based): {len(high_na)}")
        logger.info(f"Examples: {high_na[:10]}")
        X_train = X_train.drop(columns=high_na)
        X_test = X_test.drop(columns=high_na, errors="ignore")

    # 5a. Impute all remaining NaNs with -999 (matches old notebook)
    X_train = X_train.fillna(-999)
    X_test = X_test.fillna(-999)

    # 5b. Drop low-coverage categorical columns (top-5 < 90%) based on TRAIN
    cat_cols = X_train.select_dtypes(exclude=[np.number]).columns
    low_covors = []
    for c in cat_cols:
        freq_top5 = (
            X_train[c].astype(str).value_counts(normalize=True).nlargest(5).sum() * 100
        )
        if freq_top5 < 90:
            low_covors.append(c)

    if low_covors:
        logger.info(f"Dropping low-coverage categorical cols: {len(low_covors)}")
        logger.info(f"Examples: {low_covors[:10]}")
        X_train = X_train.drop(columns=low_covors)
        X_test = X_test.drop(columns=low_covors, errors="ignore")

    # 5c. One-hot encode remaining non-numeric columns on TRAIN and align TEST
    to_dummy = X_train.select_dtypes(exclude=[np.number]).columns.tolist()
    logger.info(f"One-hot encoding {len(to_dummy)} columns")
    X_train = pd.get_dummies(X_train, columns=to_dummy, drop_first=False)
    X_test = pd.get_dummies(X_test, columns=to_dummy, drop_first=False)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    # 5d. Scale ALL numeric features (including dummies) — notebook behaviour
    num_cols = X_train.select_dtypes(include=[np.number]).columns
    scaler = MinMaxScaler().fit(X_train[num_cols])
    X_train[num_cols] = scaler.transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])

    # 6. Re-attach meta
    train_proc = pd.concat(
        [meta_train.reset_index(drop=True), X_train.reset_index(drop=True)], axis=1
    )
    test_proc = pd.concat(
        [meta_test.reset_index(drop=True), X_test.reset_index(drop=True)], axis=1
    )

    # 7. Map string labels to 0/1
    mapping = {"Fully Paid": 1, "Charged Off": 0}
    train_proc[TARGET] = train_proc[TARGET].map(mapping)
    test_proc[TARGET] = test_proc[TARGET].map(mapping)

    logger.info(
        f"Processed train shape: {train_proc.shape}, processed test shape: {test_proc.shape}"
    )

    # 8. Save processed full datasets
    train_proc.to_csv(PROCESSED_TRAIN, index=False)
    test_proc.to_csv(PROCESSED_TEST, index=False)
    logger.info(f"Saved {PROCESSED_TRAIN}")
    logger.info(f"Saved {PROCESSED_TEST}")

    # 9. Save per-group train/test CSVs (for diagnostics & STRIKE groups)
    logger.info("Creating per-group train/test CSVs")
    for grp, feats in FEATURE_GROUPS.items():
        existing_feats = [f for f in feats if f in train_proc.columns]
        missing = sorted(set(feats) - set(existing_feats))
        if missing:
            logger.warning(
                f"[{grp}] Missing {len(missing)} features after preprocessing. Examples: {missing[:10]}"
            )

        cols = []
        if ID_COL in train_proc.columns:
            cols.append(ID_COL)
        cols += existing_feats
        cols.append(TARGET)

        grp_train = train_proc[cols].copy()
        grp_test = test_proc[[c for c in cols if c in test_proc.columns]].copy()

        train_path = f"{GROUP_DATA_DIR}/{grp}_train.csv"
        test_path = f"{GROUP_DATA_DIR}/{grp}_test.csv"
        grp_train.to_csv(train_path, index=False)
        grp_test.to_csv(test_path, index=False)
        logger.info(
            f"[{grp}] train: {grp_train.shape}, test: {grp_test.shape} → saved to {train_path}, {test_path}"
        )

    logger.info("✅ Preprocessing completed successfully.")


if __name__ == "__main__":
    main()
