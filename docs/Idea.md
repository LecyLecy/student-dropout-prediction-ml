# Early Student Dropout Prediction Idea

## Overview

This project predicts student dropout risk as early as possible using only enrollment/background information and academic path information that is reasonable to collect in an MVP input form.

The final direction is intentionally fixed and simple: use the same 10 MVP features from EDA through preprocessing, model training, saved artifacts, and the Streamlit app. This avoids leakage from semester performance variables and avoids confusing manual inputs such as macroeconomic indicators or dataset-specific admission codes.

## Target Transformation

The target is transformed into binary classification:

- Graduate = 0
- Dropout = 1
- Enrolled is removed

## Final MVP Features

```text
Marital status
Course
Previous qualification
Mother's qualification
Father's qualification
Displaced
Educational special needs
Gender
Age at enrollment
International
```

## Variables to Remove

The study excludes variables that are not suitable for early MVP prediction:

- Debtor
- Tuition fees up to date
- Scholarship holder
- Application mode
- Application order
- Daytime/evening attendance
- Nacionality
- Mother's occupation
- Father's occupation
- Unemployment rate
- Inflation rate
- GDP
- Curricular units 1st sem (credited)
- Curricular units 1st sem (enrolled)
- Curricular units 1st sem (evaluations)
- Curricular units 1st sem (approved)
- Curricular units 1st sem (grade)
- Curricular units 1st sem (without evaluations)
- Curricular units 2nd sem (credited)
- Curricular units 2nd sem (enrolled)
- Curricular units 2nd sem (evaluations)
- Curricular units 2nd sem (approved)
- Curricular units 2nd sem (grade)
- Curricular units 2nd sem (without evaluations)

## Models

The final project compares two classical machine learning models:

- Logistic Regression as the simple and interpretable baseline
- Random Forest as the stronger non-linear model for categorical-heavy tabular data

Naive Bayes and XGBoost are no longer part of the final MVP pipeline. This keeps the experiment focused and easier to explain while still comparing a linear baseline against a non-linear model.

## Preprocessing

The preprocessing remains straightforward:

- Remove Enrolled from the target.
- Encode target into binary classes.
- Keep only the 10 final MVP features plus target.
- Use one-hot encoding for categorical features during modeling.
- Use scaling for `Age at enrollment`.
- Evaluate models using accuracy, precision, recall, F1-score, ROC-AUC, confusion matrix, and classification report.

## Main Research Question

Can student dropout risk be predicted meaningfully using only selected early enrollment/background features, and does Random Forest outperform Logistic Regression under this MVP feature scope?

## Contribution

1. It uses a fixed, understandable MVP feature set from start to finish.
2. It removes semester academic variables and post-acceptance/admin variables to reduce leakage.
3. It avoids manual macroeconomic inputs and dataset-specific admission fields that may confuse users.
4. It compares a simple interpretable baseline with a stronger non-linear model.

## Recommended Title

Early Student Dropout Prediction Using Selected Enrollment Features: A Comparison of Logistic Regression and Random Forest
