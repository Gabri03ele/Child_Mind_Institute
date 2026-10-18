# Predicting Problematic Internet Use in Children and Adolescents

Supervised-classification project for the **Data and Web Mining** course (University of Venice), based on the Kaggle competition
[Child Mind Institute — Problematic Internet Use](https://www.kaggle.com/competitions/child-mind-institute-problematic-internet-use).

**Authors:** Gabriele Armani, Nicolò Scantamburlo

## Goal

Predict the **Severity Impairment Index (`sii`)**, a 4-class target from 0 (no problem) to 3 (severe problems), from physical-activity, body-measurement and psychological-wellbeing indicators.

## Repository contents

| File             | Description                                                                                         |
| ---------------- | --------------------------------------------------------------------------------------------------- |
| `main.ipynb`     | Full pipeline: cleaning, feature engineering, model training, tuning, feature selection, evaluation |
| `dwm_report.pdf` | Written report with methods, results and discussion                                                 |
| `dati/`          | **Not included.** Kaggle data, see below                                                            |

## Data

The data is not versioned in this repository. To run the notebook:

1. Download the data from the [competition page](https://www.kaggle.com/competitions/child-mind-institute-problematic-internet-use/data) (accepting the competition rules).
2. Create a `dati/` folder next to `main.ipynb` and put these files in it:
   `train.csv`, `test.csv`, `data_dictionary.csv`, `sample_submission.csv`.

The notebook itself only reads `dati/train.csv`.

## Method

**Preprocessing**

- Dropped rows with a missing target and all `PCIAT-*` columns (the target is derived directly from them, so they would leak it), plus the `id` column.
- Removed rows with `Physical-Weight == 0` (implausible).
- Merged `Physical-BMI` and `BIA-BIA_BMI` into a single averaged BMI feature.
- One-hot encoded the `Season` columns.
- Dropped columns with >50% missing values unless their correlation with the target was >0.10.
- Mean imputation for the tree-based models; XGBoost and LightGBM are trained on the non-imputed data.

**Models:** Decision Tree (with pruning), Random Forest, XGBoost, LightGBM.

**Common procedure for each model**

1. Default hyperparameters.
2. Tuning with `GridSearchCV` / `RandomizedSearchCV`, 20-fold cross-validation.
3. Feature selection with `RFECV`.
4. Retraining on the selected features and evaluation (accuracy, confusion matrix, feature importance).

The split is 80% train / 20% test (`random_state=42`).

## Results

The target is imbalanced (class 0: 58.1%, 1: 26.7%, 2: 13.9%, 3: 1.3%), so always predicting class 0 already gives about **58% accuracy** as baseline.

| Model         | Train acc. | Test acc. | CV acc. (mean) |
| ------------- | ---------- | --------- | -------------- |
| Decision Tree | 61.95%     | 59.78%    | 61.85%         |
| Random Forest | 65.77%     | 58.29%    | 61.90%         |
| XGBoost       | 62.65%     | 58.10%    | 61.66%         |
| LightGBM      | 62.32%     | 58.47%    | 61.85%         |

All models perform only slightly above the majority-class baseline and mostly predict class 0. The rare classes, especially class 3, are hard to detect. Features that were consistently important across models: `Basic_Demos-Age`, `Physical-Height`, `Physical-Weight`, `SDS-SDS_Total_Raw`, `PreInt_EduHx-computerinternet_hoursday`. Pruning the decision tree did not change the tree or its predictions.

## How to run

```bash
git clone <repo-url>
cd <repo-folder>
python -m venv .venv && source .venv/bin/activate
pip install numpy pandas matplotlib seaborn scikit-learn xgboost lightgbm jupyter
jupyter notebook main.ipynb
```

(Add the data to `dati/` first, as described above.) Hyperparameter searches with 20-fold CV are slow, so a full run can take a while.

## Possible improvements

- Use a metric suited to imbalance (e.g. quadratic weighted kappa, as in the competition, or macro-F1) instead of accuracy.
- Handle class imbalance (class weights, resampling).
- Use the accelerometer time-series files, which this project does not use.
