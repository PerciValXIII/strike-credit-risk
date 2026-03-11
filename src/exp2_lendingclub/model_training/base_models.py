# src/exp2_lendingclub/model_training/base_models.py

import xgboost as xgb
from lightgbm import LGBMClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
)

from .config import HYPERPARAM_TUNING, N_TRIALS

# Base configs – close to your notebook defaults (no heavy tuning by default)
MODEL_CONFIGS = {
    "XGBoost": {
        "model": xgb.XGBClassifier,
        "params": {
            "learning_rate": 0.05,
            "n_estimators": 200,
            "max_depth": 4,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "objective": "binary:logistic",
            "n_jobs": -1,
            "random_state": 42,
            "eval_metric": "logloss",
        },
        "tune_params": {
            "max_depth": (3, 10),
            "learning_rate": (0.01, 0.3),
            "n_estimators": (50, 300),
        },
    },
    "GBDT": {
        "model": GradientBoostingClassifier,
        "params": {
            "learning_rate": 0.05,
            "n_estimators": 200,
            "max_depth": 3,
            "subsample": 0.8,
            "random_state": 42,
        },
        "tune_params": {
            "max_depth": (2, 5),
            "learning_rate": (0.01, 0.3),
            "n_estimators": (50, 300),
        },
    },
    "AdaBoost": {
        "model": AdaBoostClassifier,
        "params": {"n_estimators": 200, "learning_rate": 0.05, "random_state": 42},
        "tune_params": {
            "n_estimators": (50, 300),
            "learning_rate": (0.01, 1.0),
        },
    },
    "RandomForest": {
        "model": RandomForestClassifier,
        "params": {
            "n_estimators": 200,
            "max_depth": 12,
            "max_features": 0.2,
            "min_samples_leaf": 2,
            "n_jobs": -1,
            "random_state": 42,
        },
        "tune_params": {
            "n_estimators": (50, 300),
            "max_depth": (3, 20),
        },
    },
    "LightGBM": {
        "model": LGBMClassifier,
        "params": {
            "learning_rate": 0.05,
            "n_estimators": 200,
            "max_depth": 6,
            "num_leaves": 31,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "n_jobs": -1,
            "random_state": 42,
        },
        "tune_params": {
            "learning_rate": (0.01, 0.3),
            "n_estimators": (50, 300),
            "num_leaves": (16, 64),
        },
    },
}

HYPERPARAM_TUNING_ENABLED = HYPERPARAM_TUNING
TUNING_TRIALS = N_TRIALS
