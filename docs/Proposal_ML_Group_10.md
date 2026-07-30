# Student Dropout Risk Classification Using Early Enrollment and Background Features

_Source file: Proposal ML - Group 10(2).docx_

## I. Problem

Higher-education dropout has serious consequences for individuals, institutions, and national productivity. Detection systems that rely only on current-semester grades may respond too late because those signals become available only after students have begun their studies.

This project uses the *Predict Students' Dropout and Academic Success* dataset to build a binary classifier that estimates dropout risk from information suitable for an early-stage MVP: enrollment details, prior academic pathways, and student background.

## II. Dataset

Dataset:

[Predict Students' Dropout and Academic Success | UCI](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success)

The original target has three classes: Graduate, Dropout, and Enrolled. Enrolled records are removed so the project becomes a binary classification task:

- Graduate = 0
- Dropout = 1

## III. Feature Scope

The final MVP uses a fixed set of ten features:

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

Semester academic performance, post-acceptance administrative status, macroeconomic variables, application mode and order, occupation variables, and nationality are excluded. This keeps the MVP concise and understandable while reducing leakage risk.

## IV. Modeling

The project compares five classical models:

1. Logistic Regression
2. Random Forest
3. Gradient Boosting
4. Extra Trees
5. SVM (RBF)

Logistic Regression provides a simple, interpretable baseline. Random Forest is selected as the primary model because it offers the strongest balance of dropout-class F1 score and recall. Its decision threshold is lowered to `0.40` to make the early-warning system more sensitive to at-risk students.

## V. Preprocessing

The preprocessing workflow:

- remove Enrolled records;
- encode the target as Graduate = 0 and Dropout = 1;
- save the ten MVP features and target to `data/processed/processed.csv`;
- pass binary features through unchanged;
- apply `RobustScaler` to `Age at enrollment`;
- apply `OneHotEncoder` to nominal features with manageable cardinality;
- apply `TargetEncoder` to Mother's qualification and Father's qualification.

## VI. Evaluation Metrics

The evaluation uses:

1. **F1 score** to balance precision and recall for the dropout-risk class;
2. **Recall** to reduce the number of at-risk students the model misses;
3. **Precision** to measure how often risk flags are correct;
4. **ROC–AUC** to evaluate probability-based class separation;
5. **Confusion matrix** to inspect the model's error patterns.

## VII. Deployment

The trained model is stored as a scikit-learn pipeline and served through a Streamlit application. A user completes a ten-field student profile, and the system returns an estimated dropout probability and a review-oriented risk signal.

The output is intended to support human-led outreach, not to make an automatic academic decision.

## VIII. Reference

Realinho, V., Machado, J., Baptista, J., & Martins, M. V. (2022). *Predict Students' Dropout and Academic Success* [Data set]. UCI Machine Learning Repository. https://doi.org/10.24432/C5MC89
