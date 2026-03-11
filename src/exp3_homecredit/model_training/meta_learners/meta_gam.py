import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import SplineTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


def _prob_to_logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def train_meta_model_gam(
    demog_preds,
    deq_preds,
    vin_preds,
    target_df,
    logger=None
):
    # --- Build meta dataset (same logic as LR) ---
    meta_df = demog_preds.merge(deq_preds, on="SK_ID_CURR", how="outer")
    meta_df = meta_df.merge(vin_preds, on="SK_ID_CURR", how="outer")
    meta_df = meta_df.merge(target_df, on="SK_ID_CURR", how="left")
    meta_df = meta_df.dropna()

    X = meta_df.drop(columns=["SK_ID_CURR", "TARGET"])
    y = meta_df["TARGET"]

    # --- Convert probabilities to logits ---
    X_logit = X.apply(_prob_to_logit)

    # --- GAM-style spline expansion ---
    spline = SplineTransformer(
        n_knots=5,
        degree=3,
        include_bias=False
    )
    X_spline = spline.fit_transform(X_logit)

    model = LogisticRegression(
        penalty="elasticnet",
        solver="saga",
        l1_ratio=0.3,
        max_iter=3000,
        random_state=42
    )

    scores = cross_val_score(model, X_spline, y, cv=5, scoring="roc_auc")

    if logger:
        logger.info(f"[META-GAM] AUC (CV): {scores.mean():.4f}")
    else:
        print(f"[META-GAM] AUC (CV): {scores.mean():.4f}")

    model.fit(X_spline, y)

    joblib.dump(
        {"spline": spline, "model": model},
        "outputs/models/meta_gam_model.pkl"
    )

    return model
