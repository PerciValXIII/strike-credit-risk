import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from lightgbm import LGBMClassifier

from src.exp2_lendingclub.model_training.utils_logger import get_logger


# ---------------------------------------------------------
# Path Setup
# ---------------------------------------------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))      # .../ablationA_monolithic
MT_DIR   = os.path.dirname(THIS_DIR)                        # .../model_training

# Correct location where monolithic CSVs were saved
DATA_DIR  = os.path.join(MT_DIR, "group_datasets")
MODEL_DIR = os.path.join(THIS_DIR, "models")
OOF_DIR   = os.path.join(THIS_DIR, "oof")
LOG_DIR   = os.path.join(THIS_DIR, "logs")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OOF_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

logger = get_logger(
    "ablationA_train",
    os.path.join(LOG_DIR, "train_base_models_ablationA.log")
)

NFOLDS = 5
SEED   = 42
TARGET_COL = "loan_status"
ID_COL     = "id"


# ---------------------------------------------------------
# Base Model Configurations (default STRIKE params)
# ---------------------------------------------------------
MODEL_CONFIGS = {
    "XGBoost": xgb.XGBClassifier(
        learning_rate=0.05,
        n_estimators=200,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        n_jobs=-1,
        random_state=SEED
    ),
    "GBDT": GradientBoostingClassifier(
        learning_rate=0.05,
        n_estimators=200,
        max_depth=4,
        subsample=0.8,
        random_state=SEED
    ),
    "AdaBoost": AdaBoostClassifier(
        n_estimators=200,
        learning_rate=0.05,
        random_state=SEED
    ),
    "RandomForest": RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        max_features=0.2,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=SEED
    ),
    "LightGBM": LGBMClassifier(
        learning_rate=0.05,
        n_estimators=200,
        max_depth=6,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,
        random_state=SEED
    )
}


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():
    logger.info("🚀 Ablation A: Starting monolithic base model training")

    train_path = os.path.join(DATA_DIR, "Monolithic_All_train.csv")
    df = pd.read_csv(train_path)
    logger.info(f"Loaded Monolithic_All_train.csv: {df.shape}")

    y = df[TARGET_COL].values
    X = df.drop(columns=[TARGET_COL, ID_COL])
    logger.info(f"Feature matrix: {X.shape}, Target: {y.shape}")

    kf = KFold(n_splits=NFOLDS, shuffle=True, random_state=SEED)

    oof_preds = {}
    model_aucs = {}

    # -----------------------------------------------------
    # Train all baseline models with 5-fold OOF
    # -----------------------------------------------------
    for model_name, model_obj in MODEL_CONFIGS.items():
        logger.info(f"[Monolithic_All] ▶ Training model: {model_name}")

        oof = np.zeros(len(X))

        for fold, (tr_idx, val_idx) in enumerate(kf.split(X), start=1):
            X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
            y_tr, y_val = y[tr_idx], y[val_idx]

            # fresh model instance
            mdl = model_obj.__class__(**model_obj.get_params())
            mdl.fit(X_tr, y_tr)

            fold_path = os.path.join(MODEL_DIR, f"MonolithicAll_{model_name}_fold{fold}.pkl")
            joblib.dump(mdl, fold_path)
            logger.info(f"[{model_name}] Saved model: {fold_path}")

            oof[val_idx] = mdl.predict_proba(X_val)[:, 1]

        auc = roc_auc_score(y, oof)
        model_aucs[model_name] = auc
        oof_preds[model_name] = oof

        logger.info(f"[Monolithic_All] [{model_name}] FINAL OOF AUC = {auc:.4f}")

    # -----------------------------------------------------
    # Save OOF predictions
    # -----------------------------------------------------
    oof_df = pd.DataFrame({f"{m}_oof": p for m, p in oof_preds.items()})
    oof_df[TARGET_COL] = y

    oof_out = os.path.join(OOF_DIR, "oof_predictions_all_models.csv")
    oof_df.to_csv(oof_out, index=False)
    logger.info(f"Saved OOF predictions → {oof_out}")

    # -----------------------------------------------------
    # Select global top-3 models
    # -----------------------------------------------------
    top_models = sorted(model_aucs.items(), key=lambda x: x[1], reverse=True)[:3]
    logger.info(f"🏆 Top-3 models (monolithic): {top_models}")

    # Build meta dataset
    meta_cols = [f"{m}_oof" for m, _ in top_models]
    meta_train = oof_df[meta_cols + [TARGET_COL]]

    meta_out = os.path.join(OOF_DIR, "meta_train_top_models.csv")
    meta_train.to_csv(meta_out, index=False)

    logger.info(f"Saved meta_train_top_models.csv → {meta_train.shape}")

    # Save top model info for meta training step
    top_models_path = os.path.join(OOF_DIR, "top_models_monolithic.pkl")
    joblib.dump(top_models, top_models_path)

    logger.info("✅ Ablation A base model training completed successfully.")


if __name__ == "__main__":
    main()
