# src/exp2_lendingclub/model_training/config.py

import os

# ---------- Paths ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, "..", "data", "filtered_data_lendingclub.csv")

PROCESSED_TRAIN = os.path.join(BASE_DIR, "processed_train.csv")
PROCESSED_TEST  = os.path.join(BASE_DIR, "processed_test.csv")

GROUP_DATA_DIR   = os.path.join(BASE_DIR, "group_datasets")
MODEL_DIR        = os.path.join(BASE_DIR, "models")
OOF_DIR          = os.path.join(BASE_DIR, "oof")
LOG_DIR          = os.path.join(BASE_DIR, "logs")

os.makedirs(GROUP_DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OOF_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# ---------- CV / Meta ----------
NFOLDS = 5
SEED = 42

TARGET = "loan_status"
ID_COL = "id"

# ---------- Feature Groups ----------
FEATURE_GROUPS = {
    "Loan_Terms": [
        "loan_amnt", "funded_amnt", "funded_amnt_inv",
        "int_rate", "installment", "dti", "annual_inc", "policy_code"
    ],
    "Credit_Profile": [
        "delinq_2yrs", "fico_range_low", "fico_range_high",
        "inq_last_6mths", "mths_since_last_delinq", "mths_since_last_record",
        "open_acc", "pub_rec", "total_acc", "acc_now_delinq",
        "collections_12_mths_ex_med", "mths_since_last_major_derog",
        "pub_rec_bankruptcies", "tax_liens"
    ],
    "Utilization_and_Activity": [
        "revol_bal", "revol_util", "tot_cur_bal", "tot_hi_cred_lim",
        "total_bal_il", "il_util", "total_bal_ex_mort", "total_bc_limit",
        "total_il_high_credit_limit", "bc_open_to_buy", "bc_util",
        "total_rev_hi_lim", "avg_cur_bal", "num_rev_accts",
        "num_rev_tl_bal_gt_0", "pct_tl_nvr_dlq", "percent_bc_gt_75",
        "num_accts_ever_120_pd", "num_actv_bc_tl", "num_actv_rev_tl",
        "num_bc_sats", "num_bc_tl", "num_il_tl", "num_op_rev_tl",
        "num_sats", "num_tl_120dpd_2m", "num_tl_30dpd",
        "num_tl_90g_dpd_24m", "num_tl_op_past_12m"
    ],
    "Categorical_Flags": [
        "term_ 36 months", "term_ 60 months",
        "grade_A", "grade_B", "grade_C", "grade_D", "grade_E", "grade_F", "grade_G",
        "home_ownership_ANY", "home_ownership_MORTGAGE", "home_ownership_NONE",
        "home_ownership_OWN", "home_ownership_RENT",
        "verification_status_Not Verified", "verification_status_Source Verified",
        "verification_status_Verified",
        "pymnt_plan_n", "application_type_Individual", "application_type_Joint App",
        "hardship_flag_N", "disbursement_method_Cash", "disbursement_method_DirectPay",
        "debt_settlement_flag_N", "debt_settlement_flag_Y",
        "purpose_car", "purpose_credit_card", "purpose_debt_consolidation",
        "purpose_home_improvement", "purpose_house", "purpose_major_purchase",
        "purpose_medical", "purpose_moving", "purpose_other",
        "purpose_renewable_energy", "purpose_small_business", "purpose_vacation",
        "purpose_wedding",
        "title_Business", "title_Car financing", "title_Credit card refinancing",
        "title_Debt consolidation", "title_Home buying",
        "title_Home improvement", "title_Major purchase", "title_Medical expenses",
        "title_Moving and relocation", "title_Other", "title_Vacation",
        "initial_list_status_f", "initial_list_status_w"
    ]
}

# ---------- Hyperparameter tuning flag ----------
HYPERPARAM_TUNING = False     # keep False to stay closer to “no heavy tuning” story
N_TRIALS = 10                 # used only if HYPERPARAM_TUNING=True
