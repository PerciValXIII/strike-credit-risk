# src/exp2_lendingclub/model_training/train_base_models.py

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import roc_auc_score

from .config import (
    PROCESSED_TRAIN,
    GROUP_DATA_DIR,
    FEATURE_GROUPS,
    MODEL_DIR,
    OOF_DIR,
    NFOLDS,
    SEED,
    TARGET,
)
from .utils_logger import get_logger
from .base_models import MODEL_CONFIGS, HYPERPARAM_TUNING_ENABLED, TUNING_TRIALS

try:
    import optuna
except ImportError:
    optuna = None


def tune_hyperparameters(X, y, cfg, logger):
    if not optuna:
        logger.warning("Optuna not installed; skipping tuning.")
        return {}

    def objective(trial):
        params = {}
        for p, (low, high) in cfg["tune_params"].items():
            if isinstance(low, int):
                params[p] = trial.suggest_int(p, low, high)
            else:
                params[p] = trial.suggest_float(p, low, high)
        mdl = cfg["model"](**{**cfg["params"], **params})
        scores = cross_val_score(mdl, X, y, cv=3, scoring="roc_auc")
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=TUNING_TRIALS)
    best_params = study.best_params
    logger.info(f"Best params: {best_params}")
    return best_params


def get_group_train_df(group_name: str) -> pd.DataFrame:
    path = os.path.join(GROUP_DATA_DIR, f"{group_name}_train.csv")
    return pd.read_csv(path)


def main():
    logger = get_logger("train_base_models", "train_base_models_lendingclub.log")
    logger.info("🚀 Starting STRIKE base model training for LendingClub")

    # Load full processed train (mainly for y)
    train_proc = pd.read_csv(PROCESSED_TRAIN)
    y_full = train_proc[TARGET].values
    n_samples = len(train_proc)
    logger.info(f"Loaded processed_train.csv: {train_proc.shape}")

    # Structures to store OOF predictions and performance
    all_oof_cols = []
    all_oof_preds = []
    model_perf = {grp: {} for grp in FEATURE_GROUPS.keys()}

    for grp in FEATURE_GROUPS.keys():
        logger.info(f"\n==================== Group: {grp} ====================")
        df_grp = get_group_train_df(grp)
        assert TARGET in df_grp.columns, f"{TARGET} missing in {grp} train data"
        y = df_grp[TARGET].values
        X = df_grp.drop(columns=[TARGET, "id"], errors="ignore")

        logger.info(f"[{grp}] X shape: {X.shape}, y shape: {y.shape}")

        for mname, cfg in MODEL_CONFIGS.items():
            logger.info(f"[{grp}] ▶ Training model: {mname}")

            # Optional hyperparameter tuning
            params = cfg["params"].copy()
            if HYPERPARAM_TUNING_ENABLED and "tune_params" in cfg:
                logger.info(f"[{grp}] 🔧 Tuning {mname}")
                best = tune_hyperparameters(X, y, cfg, logger)
                params.update(best)

            kf = KFold(n_splits=NFOLDS, shuffle=True, random_state=SEED)
            oof = np.zeros(len(X))

            for fold, (tr_idx, val_idx) in enumerate(kf.split(X), start=1):
                X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
                y_tr, y_val = y[tr_idx], y[val_idx]

                model = cfg["model"](**params)
                model.fit(X_tr, y_tr)

                # Save fold model
                model_path = os.path.join(
                    MODEL_DIR, f"{grp}_{mname}_fold{fold}.pkl"
                )
                joblib.dump(model, model_path)
                logger.info(f"[{grp}] [{mname}] Saved {model_path}")

                oof[val_idx] = model.predict_proba(X_val)[:, 1]

            auc = roc_auc_score(y, oof)
            logger.info(f"[{grp}] [{mname}] FINAL OOF AUC = {auc:.4f}")

            col_name = f"{grp}__{mname}"
            all_oof_cols.append(col_name)
            all_oof_preds.append(oof.reshape(-1, 1))
            model_perf[grp][mname] = auc

    # Stack all OOF predictions horizontally
    stacked_oof = np.hstack(all_oof_preds)
    oof_df = pd.DataFrame(stacked_oof, columns=all_oof_cols)
    oof_df[TARGET] = y_full
    oof_path = os.path.join(OOF_DIR, "oof_predictions_all_models.csv")
    oof_df.to_csv(oof_path, index=False)
    logger.info(f"Saved OOF predictions to {oof_path}")

    # Determine top-3 models per group
    top_models = {}
    for grp, scores in model_perf.items():
        top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
        top_models[grp] = top
        logger.info(f"[{grp}] Top-3 models: {top}")

    top_models_path = os.path.join(OOF_DIR, "top_models_per_group.pkl")
    joblib.dump(top_models, top_models_path)
    logger.info(f"Saved top_models dict to {top_models_path}")

    # Save a "meta-train" file (only top model columns)
    selected_cols = []
    for grp, models in top_models.items():
        for mname, _ in models:
            selected_cols.append(f"{grp}__{mname}")

    meta_train = oof_df[selected_cols + [TARGET]].copy()
    meta_train_path = os.path.join(OOF_DIR, "meta_train_top_models.csv")
    meta_train.to_csv(meta_train_path, index=False)
    logger.info(
        f"Saved meta_train (top models) to {meta_train_path}, shape={meta_train.shape}"
    )

    logger.info("✅ Base model training completed successfully.")


if __name__ == "__main__":
    main()
