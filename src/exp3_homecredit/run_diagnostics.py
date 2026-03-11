from __future__ import annotations
import os
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from src.exp3_homecredit.feature_engineering.preprocess_pipeline import Preprocessor
from src.utils.grouping import get_homecredit_groups
from src.diagnostics.cmi import cmi_matrix_from_groups
from src.diagnostics.hstat import hstat_matrix

# -------------------------
# Paths / constants
# -------------------------
PROJ_ROOT = Path(__file__).resolve().parents[2]
HC_DATA   = PROJ_ROOT / "src" / "exp3_homecredit" / "data"

DEMOG_CSV = HC_DATA / "application_train.csv"        # has SK_ID_CURR, TARGET
DEQ_CSV   = HC_DATA / "deq_features_level1.csv"      # has SK_ID_CURR
VIN_CSV   = HC_DATA / "vintage_features_1.csv"       # has SK_ID_CURR

OUTDIR    = PROJ_ROOT / "outputs" / "diagnostics" / "homecredit"
OUTDIR.mkdir(parents=True, exist_ok=True)

ID_COL     = "SK_ID_CURR"
TARGET_COL = "TARGET"

# -------------------------
# Helpers
# -------------------------
def _load_and_preprocess(csv_path: Path) -> pd.DataFrame:
    """
    Load raw CSV and run the project Preprocessor so that the columns
    match the feature space STRIKE actually uses (OHE, scaling, etc.).

    We KEEP ID_COL and TARGET_COL if they exist, so we can merge later.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing file: {csv_path}")
    df_raw = pd.read_csv(csv_path)
    df_proc = Preprocessor(df_raw).run()

    # Sanity: we expect all numeric after Preprocessor
    non_numeric = df_proc.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        raise ValueError(
            f"Non-numeric columns remain after preprocessing in {csv_path.name}: {non_numeric}"
        )

    # Sanity: check no NaN / inf
    if not np.all(np.isfinite(df_proc.to_numpy())):
        raise ValueError(
            f"Non-finite values remain after preprocessing in {csv_path.name}."
        )

    return df_proc


def _save_heatmap(df: pd.DataFrame, title: str, path_png: Path) -> None:
    """
    Generate a publication-quality heatmap with readable text, proper color balance,
    and formatted annotations suitable for inclusion in papers.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(6.5, 5.5), dpi=300)

    # Use a perceptually uniform colormap (e.g., "viridis" or "crest")
    cmap = sns.color_palette("viridis", as_cmap=True)

    # Draw the heatmap using seaborn
    ax = sns.heatmap(
        df,
        annot=True,                     # show values in cells
        fmt=".3f",                      # 2 decimal places
        cmap = sns.color_palette("mako", as_cmap=True),
        cbar=True,
        linewidths=0.6,
        linecolor="white",
        square=True,
        annot_kws={"fontsize": 10, "color": "white"},
        xticklabels=df.columns,
        yticklabels=df.index,
    )

    

    # Formatting titles and ticks
    ax.set_title(title, fontsize=13, pad=16, fontweight="semibold")
    ax.set_xticklabels(df.columns, fontsize=10, rotation=45, ha="right")
    ax.set_yticklabels(df.index, fontsize=10, rotation=0)

    # Adjust layout for better spacing
    plt.tight_layout(pad=1.2)

    # Save high-res PNG for Overleaf
    plt.savefig(path_png, dpi=300, bbox_inches="tight")
    plt.close()



def _merge_feature_tables(
    demog: pd.DataFrame,
    deq: pd.DataFrame,
    vin: pd.DataFrame,
    *,
    id_col: str,
) -> pd.DataFrame:
    """
    Inner-join the three processed frames on SK_ID_CURR to align samples.
    We allow duplicate column names across groups (e.g. 'AMT_ANNUITY' showing
    up in both demog and deq after preprocessing) -- if that happens, pandas
    will suffix them automatically during merge. We'll live with that, because
    downstream models just see the merged numeric dataframe.
    """
    merged = demog.merge(deq, on=id_col, how="inner", suffixes=("", "_DEQ"))
    merged = merged.merge(vin, on=id_col, how="inner", suffixes=("", "_VIN"))
    return merged


def _build_group_matrices(
    merged: pd.DataFrame,
    groups: Dict[str, List[str]],
) -> Dict[str, np.ndarray]:
    """
    Build X_by_group = {group_name: ndarray} aligned row-wise with merged.
    We assume columns in `groups` match columns in `merged`.
    """
    X_by_group = {}
    for gname, cols in groups.items():
        missing = [c for c in cols if c not in merged.columns]
        if missing:
            # If we ever hit this, it means your grouping and merge aren't consistent.
            # Usually this happens if two groups produced overlapping column names that
            # got suffixed during merge. You'd fix that upstream.
            raise ValueError(
                f"Group '{gname}' columns missing after merge: {missing[:5]} ..."
            )
        X_by_group[gname] = merged[cols].to_numpy()
    return X_by_group

def _drop_label_clones(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """
    Remove any columns that are literally the target or obvious suffixed copies
    of the target after merging (e.g. TARGET_DEQ, TARGET_VIN).
    """
    cols_to_drop = [
        c for c in df.columns
        if c == target_col
        or c.startswith(target_col + "_")
        or "target" in c.lower()  # belt-and-suspenders
    ]

    if cols_to_drop:
        print("[leakage guard] Dropping target-like columns:", cols_to_drop)

    return df.drop(columns=cols_to_drop, errors="ignore")



# -------------------------
# Main pipeline
# -------------------------
def main():
    # 1. Get final group definitions (post-Preprocessor column names)
    groups = get_homecredit_groups()  # e.g. {"demographics":[...], "delinquency":[...], "vintage":[...]}

    # 2. Load + preprocess each source table
    demog = _load_and_preprocess(DEMOG_CSV)
    deq   = _load_and_preprocess(DEQ_CSV)
    vin   = _load_and_preprocess(VIN_CSV)

    # 3. Basic checks: ID/TARGET
    if ID_COL not in demog.columns:
        raise ValueError(f"{ID_COL} not found in processed demographics table.")
    if TARGET_COL not in demog.columns:
        raise ValueError(f"{TARGET_COL} not found in processed demographics table (needed for y).")
    if ID_COL not in deq.columns or ID_COL not in vin.columns:
        raise ValueError(f"{ID_COL} must exist in deq and vin tables to merge.")

    # 4. Merge on SK_ID_CURR to align rows
    merged = _merge_feature_tables(demog, deq, vin, id_col=ID_COL)

    # 5. Extract ground-truth label y BEFORE we drop duplicates
    #    We trust the 'TARGET' from demographics as the canonical label.
    y = merged[TARGET_COL].astype(int).to_numpy()

    # 6. DROP ALL target-like columns from merged so they can't leak into X
    merged_noleak = _drop_label_clones(merged, TARGET_COL)

    # 7. Build full X (all numeric features except ID)
    drop_for_X = [ID_COL]  # we already dropped all TARGET-ish columns above
    drop_existing = [c for c in drop_for_X if c in merged_noleak.columns]
    X_full = merged_noleak.drop(columns=drop_existing, errors="ignore").copy()

    # 8. Train/test split (row-wise for now)
    X_train, X_test, y_train, y_test = train_test_split(
        X_full,
        y,
        test_size=0.30,
        stratify=y,
        random_state=42,
    )

    # 9. Fit model
    model = XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # 10. Evaluate properly
    y_pred_test = model.predict_proba(X_test)[:, 1]
    auc = float(roc_auc_score(y_test, y_pred_test))

    # 10. Prepare X_by_group for CMI (must align with y, so use merged order)
    X_by_group = _build_group_matrices(merged, groups)

    # 11. Compute Conditional Mutual Information matrix
    cmi_df = cmi_matrix_from_groups(
        X_by_group=X_by_group,
        y=y,
        pca_components=5,
        random_state=42,
        pairs_k=5,
    )

    # 12. Compute H-statistic interaction matrix on TEST SET ONLY
    #     (so we're not evaluating interactions on the same data we fit on)
    H_df = hstat_matrix(
        model=model,
        X=X_test,
        groups=groups,
        n_draws=20,
        random_state=42,
    )

    # 13. Save matrices, heatmaps, and summary
    cmi_csv = OUTDIR / "cmi_matrix.csv"
    h_csv   = OUTDIR / "hstat_matrix.csv"
    cmi_png = OUTDIR / "cmi_heatmap.png"
    h_png   = OUTDIR / "hstat_heatmap.png"
    summary_json = OUTDIR / "diagnostics_summary.json"

    cmi_df.to_csv(cmi_csv, index=True)
    H_df.to_csv(h_csv, index=True)
    _save_heatmap(cmi_df, "Conditional Mutual Information (Groups | Y)", cmi_png)
    _save_heatmap(H_df, "H-statistic Interaction Strength", h_png)

    # some quick scalar summaries for the paper appendix
    # off-diagonals only
    def _offdiag_stats(M: pd.DataFrame) -> dict:
        vals = []
        for i in range(len(M)):
            for j in range(len(M)):
                if i < j:
                    vals.append(M.iloc[i, j])
        return {
            "mean_offdiag": float(np.mean(vals)),
            "median_offdiag": float(np.median(vals)),
            "max_offdiag": float(np.max(vals)),
        }

    out_summary = {
        "auc_holdout": auc,
        "cmi_stats": _offdiag_stats(cmi_df),
        "hstat_stats": _offdiag_stats(H_df),
        "groups": {k: len(v) for k, v in groups.items()},
    }

    with open(summary_json, "w") as f:
        json.dump(out_summary, f, indent=2)

    print("====================================")
    print("Diagnostics complete.")
    print(f"AUC on holdout: {auc:.4f}")
    print("CMI off-diagonal stats:", out_summary["cmi_stats"])
    print("H-stat off-diagonal stats:", out_summary["hstat_stats"])
    print(f"Saved outputs in {OUTDIR}")
    print("====================================")


if __name__ == "__main__":
    main()
