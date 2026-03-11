# src/exp2_lendingclub/model_training/evaluate_holdout.py

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
)
import matplotlib.pyplot as plt
import seaborn as sns

from .config import (
    PROCESSED_TEST,
    GROUP_DATA_DIR,
    FEATURE_GROUPS,
    MODEL_DIR,
    OOF_DIR,
    TARGET,
    NFOLDS,
)
from .utils_logger import get_logger


PLOT_DIR = os.path.join(MODEL_DIR, "evaluation_plots")
os.makedirs(PLOT_DIR, exist_ok=True)


def load_test_group_df(group_name: str) -> pd.DataFrame:
    path = os.path.join(GROUP_DATA_DIR, f"{group_name}_test.csv")
    return pd.read_csv(path)


def main():
    logger = get_logger("evaluate_holdout", "evaluate_holdout_lendingclub.log")
    logger.info("🚀 Starting holdout evaluation")

    test_proc = pd.read_csv(PROCESSED_TEST)
    y_test = test_proc[TARGET].values
    logger.info(f"Loaded processed_test.csv: {test_proc.shape}")

    top_models_path = os.path.join(OOF_DIR, "top_models_per_group.pkl")
    top_models = joblib.load(top_models_path)
    logger.info(f"Loaded top_models: {top_models}")

    # Build stacked test predictions
    stacked_preds = []
    stacked_cols = []

    for grp in FEATURE_GROUPS.keys():
        df_grp = load_test_group_df(grp)
        Xg = df_grp.drop(columns=[TARGET, "id"], errors="ignore")

        for mname, _ in top_models[grp]:
            fold_preds = []
            for fold in range(1, NFOLDS + 1):
                model_path = os.path.join(MODEL_DIR, f"{grp}_{mname}_fold{fold}.pkl")
                model = joblib.load(model_path)
                fold_preds.append(model.predict_proba(Xg)[:, 1])

            mean_preds = np.mean(fold_preds, axis=0)
            stacked_preds.append(mean_preds.reshape(-1, 1))
            stacked_cols.append(f"{grp}__{mname}")

    X_stack_test = np.hstack(stacked_preds)
    logger.info(f"Stacked test shape: {X_stack_test.shape}")

    # Load meta-model
    meta_path = os.path.join(MODEL_DIR, "stacking_meta_model.pkl")
    meta = joblib.load(meta_path)

    prob_test = meta.predict_proba(X_stack_test)[:, 1]
    auc = roc_auc_score(y_test, prob_test)
    pr_auc = average_precision_score(y_test, prob_test)
    logger.info(f"🎯 Holdout ROC AUC: {auc:.4f}")
    logger.info(f"🎯 Holdout PR AUC:  {pr_auc:.4f}")

    # Derive threshold (You can use 0.5 or Youden's J)
    fpr, tpr, thr = roc_curve(y_test, prob_test)
    best_ix = np.argmax(tpr - fpr)
    best_thr = thr[best_ix]
    logger.info(f"Optimal threshold (Youden J): {best_thr:.4f}")

    y_pred = (prob_test >= best_thr).astype(int)

    cm = confusion_matrix(y_test, y_pred)
    cls_report = classification_report(y_test, y_pred)
    logger.info(f"Confusion matrix:\n{cm}")
    logger.info(f"Classification report:\n{cls_report}")

    # --------- Plots ----------
    plt.figure(figsize=(12, 10))

    # ROC
    plt.subplot(2, 2, 1)
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()

    # Precision-Recall
    plt.subplot(2, 2, 2)
    precision, recall, _ = precision_recall_curve(y_test, prob_test)
    plt.plot(recall, precision, label=f"PR AUC = {pr_auc:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall Curve")
    plt.legend()

    # Confusion Matrix
    plt.subplot(2, 2, 3)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")

    plt.tight_layout()
    plot_path = os.path.join(PLOT_DIR, "evaluation_holdout.png")
    plt.savefig(plot_path)
    plt.close()
    logger.info(f"Saved evaluation plot to {plot_path}")

    # Save detailed predictions
    results_df = pd.DataFrame(
        {
            "true_label": y_test,
            "predicted_prob": prob_test,
            "predicted_label": y_pred,
        }
    )
    results_path = os.path.join(PLOT_DIR, "final_predictions_holdout.csv")
    results_df.to_csv(results_path, index=False)
    logger.info(f"Saved final predictions to {results_path}")

    logger.info("✅ Holdout evaluation completed successfully.")


if __name__ == "__main__":
    main()
