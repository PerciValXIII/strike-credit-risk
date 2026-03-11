# src/exp2_lendingclub/model_training/stack_meta_model.py

import os
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score

from .config import OOF_DIR, MODEL_DIR, TARGET, SEED
from .utils_logger import get_logger


def main():
    logger = get_logger("stack_meta_model", "stack_meta_model_lendingclub.log")
    logger.info("🚀 Starting meta-model training")

    meta_train_path = os.path.join(OOF_DIR, "meta_train_top_models.csv")
    meta_train = pd.read_csv(meta_train_path)

    X = meta_train.drop(columns=[TARGET])
    y = meta_train[TARGET].values
    logger.info(f"Meta-train shape: {X.shape}")

    # Simple L1-regularized Logistic Regression (consistent with STRIKE)
    meta = LogisticRegression(
        penalty="l1", solver="liblinear", C=0.5, random_state=SEED
    )

    cv_scores = cross_val_score(meta, X, y, cv=5, scoring="roc_auc")
    logger.info(f"Meta-model CV AUC: mean={cv_scores.mean():.4f}, std={cv_scores.std():.4f}")

    meta.fit(X, y)
    train_auc = roc_auc_score(y, meta.predict_proba(X)[:, 1])
    logger.info(f"Meta-model train AUC on OOF stack: {train_auc:.4f}")

    meta_path = os.path.join(MODEL_DIR, "stacking_meta_model.pkl")
    joblib.dump(meta, meta_path)
    logger.info(f"✅ Saved meta-model to {meta_path}")


if __name__ == "__main__":
    main()
