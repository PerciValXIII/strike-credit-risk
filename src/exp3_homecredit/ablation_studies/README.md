
# Ablation Studies — Home Credit (EXP3)

This folder contains the code used to run the **ablation studies** for the Home Credit experiments in the STRIKE paper.  
These ablations are intended to isolate the effect of **feature grouping** and compare STRIKE against alternative grouping setups and a no-grouping baseline.

---

## Purpose of this Folder

The main goals of the ablation experiments are:

1. **Compare semantic grouping vs alternative grouping strategies**
2. **Compare grouping-based STRIKE vs a no-grouping baseline**
3. **Test whether performance gains come from meaningful feature grouping rather than arbitrary partitioning**

These experiments are separate from the main STRIKE training pipeline and are provided to help reviewers reproduce the reported ablation results.

---

## Folder Contents

### Core files

- **`run_ablation_grouping_logic.py`**  
  Main runner for grouping-based ablation experiments.  
  This script loads the raw feature-group datasets, applies a specified grouping strategy, materializes the grouped tables, and runs the STRIKE-style training/evaluation pipeline on top of those grouped datasets.

- **`run_ablation_no_grouping.py`**  
  Runner for the **no-grouping baseline**.  
  This experiment combines all features into a single table and evaluates a conventional non-grouped modeling setup for comparison against STRIKE.

- **`grouping_logic.py`**  
  Main grouping utility module.  
  This file contains the logic for:
  - loading the raw Home Credit group tables
  - validating schema consistency
  - merging them into a master table
  - generating grouped feature partitions under different strategies

- **`utils_logging.py`**  
  Helper utilities for experiment logging.

---

### Additional / auxiliary files

- **`grouping_logic_random.py`**  
  Auxiliary grouping implementation for a **random mixed grouping** strategy.  
  This file was used for experiments where the original feature sources (demographic / delinquency / vintage) are deliberately mixed across newly formed synthetic groups.

  Note that this overlaps conceptually with some logic in `grouping_logic.py`, but it is retained here because it corresponds to a specific random-mixed grouping design used during ablation experimentation.

- **`run_random_grouping_trials.py`**  
  Utility script for running multiple random-grouping trials across seeds.  
  This is useful if one wants to assess the stability of random grouping baselines instead of relying on a single seed.

- **`logs/`**  
  Contains experiment logs generated during ablation runs.

---

## Recommended Execution Order

To reproduce the ablation results, the recommended order is:

### 1. Run the no-grouping baseline
Use:

```bash
python src/exp3_homecredit/ablation_studies/run_ablation_no_grouping.py
````

This produces the baseline result where all features are used together without STRIKE-style grouping.

---

### 2. Run the grouping-based ablations

Use:

```bash
python src/exp3_homecredit/ablation_studies/run_ablation_grouping_logic.py
```

This runs the grouping-based ablation setup using the grouping strategies configured inside the script.

Depending on how the script is configured, this may run one or more of the following grouping strategies:

* `manual` — original semantic grouping
* `random` — random reassignment of features while preserving group sizes
* `corr` — correlation-based grouping
* `mi` — mutual-information-based grouping

Please check the script arguments or internal configuration to see which strategies are currently active.

---

### 3. (Optional) Run repeated random-grouping trials

Use:

```bash
python src/exp3_homecredit/ablation_studies/run_random_grouping_trials.py
```

This is optional and is mainly intended to evaluate variability across multiple random seeds.

---

## Grouping Strategies

The ablation code explores alternative ways of partitioning the feature space into three groups before running the STRIKE stacking pipeline.

### 1. Manual grouping

This is the original semantic grouping used by STRIKE:

* **Demographic**
* **Delinquency**
* **Vintage**

This serves as the primary grouping baseline and corresponds to the intended STRIKE design.

---

### 2. Random grouping

Implemented in `grouping_logic.py`.

Here, all features are pooled together and randomly redistributed into three groups while preserving the original group sizes.
This tests whether grouping alone is sufficient, or whether the **semantic meaning of the groups** matters.

---

### 3. Random mixed grouping

Implemented in `grouping_logic_random.py`.

This is a stronger randomization baseline in which features from each original source are first shuffled and then deliberately mixed across all newly formed groups.
As a result, each new group contains a mixture of demographic, delinquency, and vintage features.

This strategy is intended to test whether STRIKE benefits specifically from **source-coherent feature groups**, rather than from arbitrary partitions of comparable size.

---

### 4. Correlation-based grouping

Implemented in `grouping_logic.py`.

Features are grouped based on inter-feature correlation structure.
This provides an unsupervised alternative to semantic grouping.

---

### 5. Mutual-information-based grouping

Implemented in `grouping_logic.py`.

Features are ranked/grouped based on mutual information with the target.
This provides a supervised alternative grouping mechanism.

---

## Which File Should Reviewers Focus On?

For reproducibility, reviewers should primarily focus on:

* **`run_ablation_grouping_logic.py`**
* **`run_ablation_no_grouping.py`**
* **`grouping_logic.py`**

These three files cover the main ablation workflow reported in the paper.

The other files (`grouping_logic_random.py`, `run_random_grouping_trials.py`) are auxiliary and were used for additional random-grouping analyses.

---

## Expected Inputs

These scripts assume access to the processed/raw Home Credit feature-group tables used in EXP3, typically corresponding to:

* demographic feature table
* delinquency feature table
* vintage feature table

Each table is expected to contain at least:

* `SK_ID_CURR`
* `TARGET`

along with the group-specific feature columns.

Please ensure the input paths inside the runner scripts are correctly set before execution.

---

## Expected Outputs

Running the ablation scripts typically produces:

* experiment logs in `logs/`
* printed evaluation metrics
* grouped intermediate tables (if the runner is configured to save them)
* model outputs / predictions depending on the underlying STRIKE training pipeline configuration

The exact save locations depend on the paths hardcoded inside each runner.

---

## Important Notes

1. **`grouping_logic.py` and `grouping_logic_random.py` are both kept intentionally.**
   They represent two related but distinct grouping implementations:

   * one for the main ablation grouping framework
   * one for the deliberately mixed random grouping setup

2. **No further code cleanup was performed here in order to preserve the exact code used for experiments.**
   The goal of this README is to make the current structure understandable and reproducible without modifying the codebase.

3. If a reviewer wants the main ablation results corresponding to the paper, they should start with:

   * `run_ablation_no_grouping.py`
   * `run_ablation_grouping_logic.py`

---

## Minimal Reproduction Checklist

1. Ensure all required Home Credit input CSV files are available in the configured paths
2. Run:

   ```bash
   python src/exp3_homecredit/ablation_studies/run_ablation_no_grouping.py
   ```
3. Run:

   ```bash
   python src/exp3_homecredit/ablation_studies/run_ablation_grouping_logic.py
   ```
4. Check generated logs under:

   ```bash
   src/exp3_homecredit/ablation_studies/logs/
   ```

---

## Contact with Main EXP3 Pipeline

These ablation scripts are designed to work alongside the broader `exp3_homecredit` experiment pipeline, especially the feature engineering, model training, and evaluation code in neighboring folders.

They should therefore be interpreted as **experiment-specific wrappers around the main STRIKE pipeline**, rather than as a fully standalone project.

---

