import os
import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import roc_auc_score

from src.exp2_lendingclub.model_training.utils_logger import get_logger

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))      # .../ablationA_monolithic
OOF_DIR  = os.path.join(THIS_DIR, "oof")
MODEL_DIR = os.path.join(THIS_DIR, "models")
LOG_DIR   = os.path.join(THIS_DIR, "logs")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(OOF_DIR, exist_ok=True)

logger = get_logger("ablationA_meta", os.path.join(LOG_DIR, "stack_meta_model_ablationA.log"))

TARGET_COL = "loan_status"
SEED = 42
NFOLDS = 5


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():
    logger.info("🚀 Ablation A: Starting meta-model training")

    meta_path = os.path.join(OOF_DIR, "meta_train_top_models.csv")
    meta_df = pd.read_csv(meta_path)
    logger.info(f"Meta-train shape: {meta_df.shape}")

    y = meta_df[TARGET_COL].values
    X = meta_df.drop(columns=[TARGET_COL]).values

    # Meta-model (same settings as STRIKE)
    meta_model = LogisticRegression(
        penalty="l1",
        solver="liblinear",
        C=0.1,
        random_state=SEED
    )

    cv = KFold(n_splits=NFOLDS, shuffle=True, random_state=SEED)
    cv_scores = cross_val_score(meta_model, X, y, cv=cv, scoring="roc_auc")
    logger.info(f"Meta-model CV AUC: mean={cv_scores.mean():.4f}, std={cv_scores.std():.4f}")

    meta_model.fit(X, y)
    oof_auc = roc_auc_score(y, meta_model.predict_proba(X)[:, 1])
    logger.info(f"Meta-model train AUC on OOF stack: {oof_auc:.4f}")

    out_path = os.path.join(MODEL_DIR, "stacking_meta_model.pkl")
    joblib.dump(meta_model, out_path)
    logger.info(f"✅ Saved meta-model to {out_path}")


if __name__ == "__main__":
    main()
