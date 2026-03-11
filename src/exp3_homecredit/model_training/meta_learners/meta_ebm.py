import numpy as np
import pandas as pd
import joblib
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.model_selection import cross_val_score


def _prob_to_logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def train_meta_model_ebm(
    demog_preds,
    deq_preds,
    vin_preds,
    target_df,
    logger=None
):
    meta_df = demog_preds.merge(deq_preds, on="SK_ID_CURR", how="outer")
    meta_df = meta_df.merge(vin_preds, on="SK_ID_CURR", how="outer")
    meta_df = meta_df.merge(target_df, on="SK_ID_CURR", how="left")
    meta_df = meta_df.dropna()

    X = meta_df.drop(columns=["SK_ID_CURR", "TARGET"])
    y = meta_df["TARGET"]

    X_logit = X.apply(_prob_to_logit)

    # Indices:
    # demog = 0,1,2 | deq = 3,4,5 | vin = 6,7,8
    interactions = [(3, 6), (4, 7), (5, 8)]

    model = ExplainableBoostingClassifier(
        interactions=interactions,
        learning_rate=0.01,
        max_bins=64,
        random_state=42
    )

    scores = cross_val_score(model, X_logit, y, cv=5, scoring="roc_auc")

    if logger:
        logger.info(f"[META-EBM] AUC (CV): {scores.mean():.4f}")
    else:
        print(f"[META-EBM] AUC (CV): {scores.mean():.4f}")

    model.fit(X_logit, y)

    joblib.dump(
        model,
        "outputs/models/meta_ebm_model.pkl"
    )

    return model
