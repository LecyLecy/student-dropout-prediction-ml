# Final Feature and Model Selection Plan

## Final MVP Features

The project uses one fixed set of 10 MVP features from EDA through preprocessing, model training, saved artifacts, and the Streamlit app:

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

These features are kept because they are available early, understandable for users, and supported by visible EDA patterns.

## Feature Rationale

- `Course` is kept because dropout rates differ clearly across study programs. Programs such as Computer Science, Equinculture, Management Evening Program, Basic Education, and Agronomy show higher dropout risk in the EDA.
- `Previous qualification` is kept because previous academic background shows meaningful dropout-rate differences across categories.
- `Mother's qualification` and `Father's qualification` are kept because family educational background shows visible differences in dropout patterns.
- `Gender` is kept because the EDA shows a clear difference in dropout proportion. This must be interpreted as association, not causation.
- `Age at enrollment` is kept as the main continuous feature because older enrollment age shows a higher dropout tendency and wider spread.
- `Marital status`, `Displaced`, `Educational special needs`, and `International` are kept as early background variables that are available before semester performance exists.

## Excluded Feature Rationale

- Semester academic performance variables are excluded because they are not available at the early prediction stage and would create leakage.
- `Debtor`, `Tuition fees up to date`, and `Scholarship holder` are excluded because they are post-acceptance or administrative status variables.
- `Unemployment rate`, `Inflation rate`, and `GDP` are excluded from the MVP input because regular users are unlikely to know the correct values. In a production system, these should be retrieved automatically from official statistics.
- `Application mode` and `Application order` are excluded because they depend on the original dataset admission context and may confuse Indonesian users.
- Occupation variables are excluded to keep the MVP input form shorter and easier to interpret.

## Model Selection

The final experiment compares five classical machine learning models:

1. Logistic Regression
2. Random Forest
3. Gradient Boosting
4. Extra Trees
5. SVM (RBF)

Logistic Regression is the baseline because it is simple, interpretable, and suitable for binary classification. Random Forest is the selected primary model because it gives the strongest Dropout F1-score in 5-fold cross-validation and works well with categorical-heavy tabular data.

The final Random Forest uses a `0.40` decision threshold for Dropout to prioritize recall in the early-warning setting. Naive Bayes is not used because earlier results were weak. XGBoost is also not used in the MVP pipeline to keep the implementation lightweight and focused on scikit-learn.

## Preprocessing Pipeline

- Binary flags (`Displaced`, `Educational special needs`, `Gender`, `International`) use passthrough because they are already 0/1.
- `Age at enrollment` uses `RobustScaler` because EDA shows right skew and high-age outliers.
- `Marital status`, `Course`, and `Previous qualification` use `OneHotEncoder(drop="first")` because they are nominal with manageable cardinality.
- `Mother's qualification` and `Father's qualification` use `TargetEncoder(cv=5)` because they have many sparse categories.
- Class balancing is used where supported because Dropout recall is the priority.

## Pipeline Summary

```text
EDA -> preprocessing -> model training -> saved model -> Streamlit app
```

- Preprocessing removes `Enrolled`.
- Target is encoded as `Graduate = 0` and `Dropout = 1`.
- `processed.csv` contains 10 features plus the encoded target.
- `mvp_features_readable.csv` contains the same 10 features plus readable `Graduate` and `Dropout` targets for inspection.
- `mvp_features_numeric.csv` contains the same 10 features in original numeric encoding plus readable target labels.
- The saved model metadata stores the fixed MVP feature list, final threshold, model comparison results, and feature groups.

More detailed reasoning and final conclusions are documented in:

```text
docs/Idea_And_Conclussion.md
```
