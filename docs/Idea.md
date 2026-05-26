# Early Student Dropout Prediction Idea

_Source file: Idea(1).docx_

## Overview

A research paper about early student dropout prediction using the *Predict Students' Dropout and Academic Success* dataset. The main goal is to predict dropout as early as possible by using only variables that are already available before the student is formally accepted into higher education.

The key idea of the paper is to compare Gaussian Naive Bayes and XGBoost under a strict pre-acceptance setting. Unlike many previous studies that rely on semester-based academic performance variables or later administrative status variables, this study will focus only on information that can realistically be known during the application or admission stage.

## Target Transformation

The target should be transformed into binary classification:

- Graduate = 0
- Dropout = 1
- Enrolled should be removed

## Variables to Remove

The study should exclude every variable that is not clearly available before student acceptance. This means the following variables should be removed:

- Debtor
- Tuition fees up to date
- Scholarship holder
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

## Valid Pre-Acceptance Variables

The remaining variables should be treated as valid pre-acceptance variables:

- Marital status
- Application mode
- Application order
- Course
- Previous qualification
- Mother's qualification
- Father's qualification
- Displaced
- Educational special needs
- Gender
- Age at enrollment
- International
- Unemployment rate
- Inflation rate
- GDP

## Feature Settings

To make the study more meaningful without making it too complicated, the experiment should use two feature settings.

### Setting A: Background-Only Variables

These variables represent demographic, family, social, and macroeconomic information that can be viewed as background characteristics:

- Marital status
- Mother's qualification
- Father's qualification
- Displaced
- Educational special needs
- Gender
- Age at enrollment
- International
- Unemployment rate
- Inflation rate
- GDP

### Setting B: Background + Application-Related Variables

This setting keeps all variables from Setting A and adds variables that are directly available from the application process:

- Application mode
- Application order
- Course
- Previous qualification

## Models

The two models used in the study should remain simple and manageable:

- Gaussian Naive Bayes as the classical probabilistic baseline
- XGBoost as the stronger boosting-based model

## Preprocessing

The preprocessing should remain straightforward:

- Remove Enrolled from the target.
- Encode target into binary classes.
- Apply the selected features only.
- Use an 80:20 train-test split.
- Apply one-hot encoding for categorical features where needed.
- Apply scaling for numerical features where needed.
- Use SMOTE on the training set if class imbalance handling is still required.
- Evaluate the models using accuracy, precision, recall, F1-score, and confusion matrix.

## Main Research Question

Can student dropout be predicted meaningfully using only pre-acceptance variables, and does XGBoost outperform Gaussian Naive Bayes under this stricter early-stage setting?

## Contribution

The contribution of the paper should be framed as follows:

1. It provides a controlled comparison between Gaussian Naive Bayes and XGBoost for early dropout prediction.
2. It uses a stricter pre-acceptance feature definition rather than the broader and more ambiguous label of non-academic variables.
3. It compares two feature settings: background-only and background plus application-related variables.
4. It evaluates whether meaningful dropout prediction is possible before semester-based academic or post-acceptance variables become available.

## Value of the Idea

The value of this idea is that it is still simple enough for a course project, but more useful and cleaner than a basic model comparison. The paper is not only asking which model performs better, but also whether institutions can already identify at-risk students using only data available before admission is finalized.

## Recommended Title

Early Student Dropout Prediction Using Pre-Acceptance Variables: A Comparison of Gaussian Naive Bayes and XGBoost
