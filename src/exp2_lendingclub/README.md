# STRIKE Experiment: LendingClub Dataset

This module applies STRIKE to the **LendingClub consumer lending dataset**, a U.S.-based peer-to-peer lending dataset with over 225,000 loans and 150 features per borrower. The dataset has moderate class imbalance (~21% defaults).

---

## Contents

- `lending_clubv2.ipynb`: Full STRIKE pipeline for this dataset.
- `__init__.py`: Module declaration.

---

## Goals

- Validate STRIKE in a real-world consumer lending context.
- Test scalability with larger dataset size and moderately imbalanced labels.

---

Model-specific metrics are discussed in the paper's Section 4.5.