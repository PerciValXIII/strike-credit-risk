import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)

from src.exp2_lendingclub.model_training.utils_logger import get_logger

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
MT_DIR   = os.path.dirname(THIS_DIR)

DATA_DIR  = os.path.join(THIS_DIR, "data")
OOF_DIR   = os.path.join(THIS_DIR, "oof")
MODEL_DIR = os.path.join(THIS_DIR, "models")
EVAL_DIR  = os.path.join(THIS_DIR, "evaluation")
LOG_DIR   = os.path.join(THIS_DIR, "logs")

os.makedirs(EVAL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OOF_DIR, exist_ok=True)

logger = get_logger("ablationA_eval", os.path.join(LOG_DIR, "evaluate_holdout_ablationA.log"))

TARGET_COL = "loan_status"
ID_COL     = "id"
NFOLDS = 5


def compute_youden_threshold(y_true, y_prob):
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    j_scores = tpr - fpr
    idx = np.argmax(j_scores)
    return thr[idx]


def main():
    logger.info("🚀 Ablation A: Starting holdout evaluation")

    test_path = os.path.join(DATA_DIR, "Monolithic_All_test.csv")
    df_test = pd.read_csv(test_path)
    logger.info(f"Loaded Monolithic_All_test.csv: {df_test.shape}")

    y_test = df_test[TARGET_COL].values
    X_test = df_test.drop(columns=[TARGET_COL, ID_COL])
    feature_cols = X_test.columns.tolist()

    # Load top models
    top_models_path = os.path.join(OOF_DIR, "top_models_monolithic.pkl")
    top_models = joblib.load(top_models_path)
    logger.info(f"Loaded top_models: {top_models}")

    # Build stacked test predictions
    stacked_preds = []

    for model_name, _ in top_models:
        logger.info(f"Generating test preds for model: {model_name}")
        fold_preds = []

        for fold in range(1, NFOLDS + 1):
            model_path = os.path.join(MODEL_DIR, f"MonolithicAll_{model_name}_fold{fold}.pkl")
            mdl = joblib.load(model_path)
            fold_preds.append(mdl.predict_proba(X_test)[:, 1])

        mean_preds = np.mean(fold_preds, axis=0)
        stacked_preds.append(mean_preds.reshape(-1, 1))

    X_stack_test = np.hstack(stacked_preds)
    logger.info(f"Stacked test shape: {X_stack_test.shape}")

    # Load meta-model
    meta_path = os.path.join(MODEL_DIR, "stacking_meta_model.pkl")
    meta_model = joblib.load(meta_path)

    y_prob = meta_model.predict_proba(X_stack_test)[:, 1]

    roc = roc_auc_score(y_test, y_prob)
    pr  = average_precision_score(y_test, y_prob)
    thr = compute_youden_threshold(y_test, y_prob)
    y_pred = (y_prob >= thr).astype(int)

    logger.info(f"🎯 Ablation A Holdout ROC AUC: {roc:.4f}")
    logger.info(f"🎯 Ablation A Holdout PR AUC:  {pr:.4f}")
    logger.info(f"Optimal threshold (Youden J): {thr:.4f}")

    cm = confusion_matrix(y_test, y_pred)
    logger.info("Confusion matrix:\n" + np.array2string(cm))

    cls_rep = classification_report(y_test, y_pred)
    logger.info("Classification report:\n" + cls_rep)

    # Plots
    plt.figure(figsize=(12, 5))

    # ROC
    plt.subplot(1, 2, 1)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, label=f"AUC = {roc:.3f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Ablation A)")
    plt.legend()

    # PR
    plt.subplot(1, 2, 2)
    prec, rec, _ = precision_recall_curve(y_test, y_prob)
    plt.plot(rec, prec, label=f"PR AUC = {pr:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall Curve (Ablation A)")
    plt.legend()

    plt.tight_layout()
    fig_path = os.path.join(EVAL_DIR, "ablationA_evaluation.png")
    plt.savefig(fig_path)
    plt.close()
    logger.info(f"Saved evaluation plot to {fig_path}")

    # Save predictions
    out_pred = pd.DataFrame({
        ID_COL: df_test[ID_COL],
        "true_label": y_test,
        "predicted_prob": y_prob,
        "predicted_label": y_pred
    })
    pred_path = os.path.join(EVAL_DIR, "ablationA_predictions_holdout.csv")
    out_pred.to_csv(pred_path, index=False)
    logger.info(f"Saved final predictions to {pred_path}")
    logger.info("✅ Ablation A holdout evaluation completed successfully.")


if __name__ == "__main__":
    main()
