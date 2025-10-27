# src/diagnostics/hstat.py
from __future__ import annotations
from typing import Dict, List, Optional
import numpy as np
import pandas as pd


def _predict_prob(model, X: pd.DataFrame) -> np.ndarray:
    """
    Get a 1D prediction score from the model.
    We assume classifier-like models. Falls back to .predict() if no proba.
    """
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    else:
        # e.g. regressor or calibrated scorer
        pred = model.predict(X)
        # ensure 1D
        return np.asarray(pred).ravel()


def _partial_dependence_group(
    model,
    X: pd.DataFrame,
    group_cols: List[str],
    *,
    baseline_X: Optional[pd.DataFrame] = None,
    n_draws: int = 20,
    random_state: int = 42,
) -> np.ndarray:
    """
    Approximate f_g(x^(g)) = E_{X^{-g}}[ f(x^(g), X^{-g}) ] for each row in X.

    For each draw:
      - Keep the group's columns from the real row.
      - Replace all NON-group columns with values from a randomly sampled row
        (this is like marginalizing out everything else).
      - Predict with the model.
    Average predictions across draws.

    Returns: np.ndarray of shape (n_rows,)
    """
    rng = np.random.default_rng(random_state)

    # columns not in the group
    other_cols = [c for c in X.columns if c not in group_cols]

    if baseline_X is None:
        # We'll sample the "other" columns from the same X
        baseline_X = X[other_cols]

    # We'll build multiple "hybrid" datasets and average predictions
    preds_all = []

    for _ in range(n_draws):
        # sample random rows for the "other" columns
        sampled_other = baseline_X.sample(
            n=len(X),
            replace=True,
            random_state=int(rng.integers(0, 1_000_000_000)),
        ).reset_index(drop=True)

        # keep actual group cols from X
        group_slice = X[group_cols].reset_index(drop=True)

        # stitch them back together in the original column order
        X_hybrid = pd.concat([group_slice, sampled_other], axis=1)
        X_hybrid = X_hybrid[X.columns]  # reorder columns to match training
        preds_all.append(_predict_prob(model, X_hybrid))

    # average predictions over draws
    preds_all = np.vstack(preds_all)  # (n_draws, n_rows)
    return preds_all.mean(axis=0)     # (n_rows,)


def hstat_group_pair(
    model,
    X: pd.DataFrame,
    group_a_cols: List[str],
    group_b_cols: List[str],
    *,
    n_draws: int = 20,
    random_state: int = 42,
) -> float:
    """
    Compute Friedman-style H^2 interaction statistic for two groups (A,B):

    H^2 = Var( f(x) - f_a(x^a) - f_b(x^b) ) / Var( f(x) )

    where:
      f(x)   = model prediction on actual full X
      f_a    = partial dependence of group A alone
      f_b    = partial dependence of group B alone

    Returns a scalar in [0,1]-ish. Higher = stronger interaction.
    """
    # full prediction
    fx = _predict_prob(model, X)

    # marginal contributions
    fa = _partial_dependence_group(
        model,
        X,
        group_a_cols,
        n_draws=n_draws,
        random_state=random_state,
    )
    fb = _partial_dependence_group(
        model,
        X,
        group_b_cols,
        n_draws=n_draws,
        random_state=random_state + 1,  # slight offset so samples differ
    )

    # numerator: variance of residual after removing additive parts
    resid = fx - fa - fb
    numer = np.var(resid)

    # denominator: total variance of model predictions
    denom = np.var(fx) + 1e-12  # avoid divide-by-zero in pathological case

    # clip to [0,1] just for numerical sanity/interpretability
    H2 = float(np.clip(numer / denom, 0.0, 1.0))
    return H2


def hstat_matrix(
    model,
    X: pd.DataFrame,
    groups: Dict[str, List[str]],
    *,
    n_draws: int = 20,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Build a symmetric matrix H where H[g,h] = H^2 interaction strength
    between group g and group h, estimated off the provided model and data.

    Steps:
    - Loop over all group pairs (g,h)
    - Compute H^2_{g,h}
    - Put results in DataFrame (groups x groups)

    Typically you call this on a held-out (test/OOF) set to avoid bias.
    """
    group_names = list(groups.keys())
    G = len(group_names)

    H = np.zeros((G, G), dtype=float)

    for i, gi in enumerate(group_names):
        for j, gj in enumerate(group_names):
            if i == j:
                H[i, j] = 0.0
                continue
            # only compute upper triangle, mirror to lower
            if j < i:
                H[i, j] = H[j, i]
                continue

            ga_cols = groups[gi]
            gb_cols = groups[gj]

            # safety: make sure all requested cols exist in X
            for col in ga_cols:
                if col not in X.columns:
                    raise ValueError(f"Column '{col}' from group '{gi}' not in X.")
            for col in gb_cols:
                if col not in X.columns:
                    raise ValueError(f"Column '{col}' from group '{gj}' not in X.")

            H_ij = hstat_group_pair(
                model,
                X,
                ga_cols,
                gb_cols,
                n_draws=n_draws,
                random_state=random_state,
            )
            H[i, j] = H_ij
            H[j, i] = H_ij

    return pd.DataFrame(H, index=group_names, columns=group_names)
