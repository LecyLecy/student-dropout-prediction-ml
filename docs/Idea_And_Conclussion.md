# Early Student Dropout Prediction Idea and Conclusion

## Project Direction

This project predicts student dropout risk as early as possible using only enrollment, background, and academic path information that is reasonable to ask in an MVP input form.

The final pipeline uses the same 10 MVP features from EDA through preprocessing, model training, saved artifacts, and the Streamlit app. This keeps the project consistent and avoids leakage from semester performance, payment status, or other post-acceptance variables.

## Target Transformation

The original target has three classes:

- Graduate
- Dropout
- Enrolled

`Enrolled` is removed because the final MVP is a binary classification task:

- Graduate = 0
- Dropout = 1

Dropout is treated as the positive class because the main goal is early risk detection.

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

These features are kept because they are available early, understandable for users, and supported by visible EDA patterns.

## Feature Rationale From EDA

`Course` is kept because dropout rates differ clearly across study programs. The EDA shows several programs with higher dropout proportions, including Computer Science, Equinculture, Management Evening Program, Basic Education, and Agronomy.

`Previous qualification` is kept because students from different previous education backgrounds show different dropout rates. This suggests that pre-university academic background provides useful early signal.

`Mother's qualification` and `Father's qualification` are kept because parental education categories show visible differences in dropout patterns. These features represent family educational background, which can support the prediction when combined with other variables.

`Gender` is kept because the EDA shows a clear difference in dropout proportion between male and female students. This is used as an observed association, not as a causal explanation.

`Age at enrollment` is kept because older enrollment age has a visible relationship with dropout. The EDA shows a right-skewed age distribution and wider spread for dropout students.

`Marital status`, `Displaced`, `Educational special needs`, and `International` are kept because they are early background variables. Some show smaller individual patterns, but they are still practical and may add useful context when combined with other features.

## Removed Variables

The following variables are excluded from the MVP scope:

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
- Curricular units 1st sem variables
- Curricular units 2nd sem variables

Semester academic variables are removed because they are not available at the early prediction stage and would create leakage. Payment and scholarship variables are also avoided because they are post-acceptance or administrative status variables.

Macroeconomic variables are excluded because regular users are unlikely to know the correct unemployment rate, inflation rate, or GDP for the enrollment context. In a production system, those values should be retrieved automatically from official statistics if they are used.

Application mode and application order are excluded because they are tied to the original dataset's admission system and may confuse users in the Indonesian MVP context.

## Preprocessing Rationale

The preprocessing pipeline is based on the EDA characteristics of each feature group.

Binary flags use passthrough:

- Displaced
- Educational special needs
- Gender
- International

These features are already encoded as 0/1, so additional scaling or encoding is unnecessary.

`Age at enrollment` uses `RobustScaler` because the EDA shows a right-skewed distribution with high-age outliers. RobustScaler uses the median and interquartile range, making it more resistant to outliers than StandardScaler.

`Marital status`, `Course`, and `Previous qualification` use `OneHotEncoder(drop="first")` because they are nominal categorical features. Their categories do not have a true numeric order, and their cardinality is still manageable for one-hot encoding.

`Mother's qualification` and `Father's qualification` use `TargetEncoder(cv=5)` because they have many categories and several sparse groups. One-hot encoding would create many low-density columns, while cross-fitted target encoding keeps the representation compact and reduces leakage risk during training.

Class balancing is used where supported because the problem is not only about overall accuracy. The project cares more about identifying Dropout students early, so Dropout recall and F1-score are more important than raw accuracy.

## Model Rationale

The project compares five classical machine learning models:

- Logistic Regression
- Random Forest
- Gradient Boosting
- Extra Trees
- SVM (RBF)

Logistic Regression is included as the interpretable baseline. It is simple, fast, and useful for checking whether the selected early features already contain meaningful predictive signal.

Random Forest and Extra Trees are included because the dataset is tabular and contains many categorical-style variables. Tree ensembles can capture non-linear relationships and feature interactions better than a purely linear model.

Gradient Boosting is included as another non-linear comparison model. It can perform well when the goal is stronger predictive performance, especially when patterns are sequentially corrected across trees.

SVM (RBF) is included as a non-linear margin-based model. It provides a different comparison point from tree-based models, especially for decision boundaries that may not be linear.

Naive Bayes is not used in the final pipeline because earlier results were weak. XGBoost is also not used in the MVP pipeline to keep the implementation lightweight and focused on scikit-learn.

## Model Results

The models are evaluated using 5-fold stratified cross-validation. Dropout is the positive class.

| Model | Recall | Precision | F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|
| Random Forest | 0.684 | 0.614 | 0.647 | 0.770 |
| Extra Trees | 0.690 | 0.599 | 0.641 | 0.764 |
| SVM (RBF) | 0.683 | 0.598 | 0.638 | 0.762 |
| Logistic Regression | 0.666 | 0.610 | 0.637 | 0.759 |
| Gradient Boosting | 0.562 | 0.665 | 0.609 | 0.766 |

Because the project is an early-warning system, the most important metric is Dropout recall, followed by F1-score. Recall matters because missing a real dropout-risk student is more harmful than giving a false warning that can still be reviewed by a human.

Random Forest is selected as the best model for this context because it has the strongest F1-score while keeping competitive recall and ROC-AUC. Extra Trees has slightly higher recall in cross-validation, but Random Forest gives the better overall recall-precision balance.

## Final Threshold Decision

The final Random Forest uses a Dropout decision threshold of `0.40` instead of the default `0.50`.

Final test result:

| Model | Threshold | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Random Forest | 0.40 | 0.654 | 0.537 | 0.838 | 0.655 | 0.775 |

This threshold is chosen because it substantially increases Dropout recall. For the MVP context, this is preferred because the app is intended to flag potential risk early, not to make a final administrative decision.

## Final Conclusion

The final project direction is:

```text
EDA -> preprocessing -> model training -> saved model -> Streamlit app
```

with one fixed feature scope:

```text
10 selected early enrollment/background features
```

and one selected primary model:

```text
Random Forest with threshold = 0.40
```

This setup is the most defendable for the MVP because it avoids leakage, keeps user input understandable, uses preprocessing that matches the EDA findings, and prioritizes Dropout recall for early intervention.

## Recommended Title

Early Student Dropout Prediction Using Selected Enrollment Features and Threshold-Tuned Random Forest
