# Student Dropout Prediction ML

A machine learning final project for predicting student dropout risk using selected early non-academic and enrollment-related features.

The project includes data preprocessing, exploratory data analysis, model training, MVP feature selection, and a Streamlit-based prediction app.

## Project Overview

This project predicts whether a student is more likely to:

- Graduate
- Dropout

The original dataset contains three target classes:

- Graduate
- Dropout
- Enrolled

For this project, `Enrolled` records are removed because the final task is binary classification between `Graduate` and `Dropout`.

The final MVP uses selected input features to keep the prediction form simple and usable.

## Dataset

Dataset used:

- Higher Education Predictors of Student Retention
- Predict Students' Dropout and Academic Success

Source:

```text
https://www.kaggle.com/datasets/thedevastator/higher-education-predictors-of-student-retention
https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success
```

## Project Structure

```text
student-dropout-prediction-ml
├── app
│   ├── app.py
│   └── feature_config.json
├── data
│   ├── raw
│   │   └── dataset.csv
│   └── processed
│       └── processed.csv
├── models
│   ├── final_mvp_model.pkl
│   ├── model_metadata.json
│   └── mvp_features.json
├── notebooks
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_model_training_and_feature_selection.ipynb
├── reports
│   └── final_model_comparison.csv
├── requirements.txt
├── README.md
└── .gitignore
```

## Main Workflow

### 1. Exploratory Data Analysis

Notebook:

```text
notebooks/01_eda.ipynb
```

Main steps:

- Load raw dataset
- Check dataset shape, columns, missing values, and duplicates
- Remove `Enrolled` records for binary EDA scope
- Remove academic performance features for early prediction scope
- Analyze target distribution
- Analyze continuous and categorical-like features
- Check feature relationships using visual and non-visual EDA

### 2. Preprocessing

Notebook:

```text
notebooks/02_preprocessing.ipynb
```

Main steps:

- Remove `Enrolled` records
- Remove academic performance features
- Encode target:
  - Graduate = 0
  - Dropout = 1
- Save processed dataset to:

```text
data/processed/processed.csv
```

### 3. Model Training and Feature Selection

Notebook:

```text
notebooks/03_model_training_and_feature_selection.ipynb
```

Main steps:

- Load processed dataset
- Split data into train, validation, and test sets
- Build preprocessing pipeline
- Train baseline models
- Evaluate models using classification metrics
- Select the best model
- Use permutation importance for MVP feature selection
- Retrain final MVP model
- Save model artifacts

### 4. Streamlit App

Main app:

```text
app/app.py
```

The app:

- Loads the final trained MVP model
- Loads feature configuration
- Displays user-friendly input labels
- Converts text input options back into encoded values for the model
- Predicts Graduate or Dropout risk
- Shows prediction probabilities

## Final MVP Features

The MVP uses selected features that are important and practical for user input.

Current MVP features:

```text
Tuition fees up to date
Course
Scholarship holder
Gender
Debtor
Mother's qualification
Mother's occupation
Age at enrollment
```

Some features were excluded for usability reasons:

- `Unemployment rate` is a macroeconomic indicator and should ideally be filled automatically from official statistics, not manually by users.
- `Application mode` is based on the original dataset's admission system and may be confusing for Indonesian users.

## Model Artifacts

Saved model files:

```text
models/final_mvp_model.pkl
models/model_metadata.json
models/mvp_features.json
```

These files are included in the repository so the Streamlit app can run without retraining the model.

## Requirements

Main libraries:

```text
pandas
numpy
scikit-learn
matplotlib
streamlit
joblib
jupyter
ipykernel
```

Install all dependencies using:

```bash
pip install -r requirements.txt
```

## Setup Instructions

### 1. Check Existing Installation

Make sure Python, Streamlit, Jupyter, and Conda are available.

```bash
python --version
streamlit --version
jupyter --version
conda --version
```

### 2. Create Conda Environment

Create a dedicated Conda environment for this project.

```bash
conda create -n student_dropout_ml python=3.11 -y
```

Activate the environment.

```bash
conda activate student_dropout_ml
```

Install dependencies.

```bash
pip install -r requirements.txt
```

### 3. Register Jupyter Kernel

Register the Conda environment as a Jupyter kernel.

```bash
python -m ipykernel install --user --name student_dropout_ml --display-name "Python (student_dropout_ml)"
```

After this, select this kernel in VS Code or Jupyter Notebook:

```text
Python (student_dropout_ml)
```

### 4. Run the Streamlit App

Run the app from the project root folder.

Recommended command:

```bash
conda activate student_dropout_ml
python -m streamlit run app/app.py
```

Alternative command:

```bash
streamlit run app/app.py
```

If the app opens successfully, Streamlit will show a local URL similar to:

```text
http://localhost:8501
```

## Running the Notebooks

Run notebooks in this order:

```text
1. notebooks/01_eda.ipynb
2. notebooks/02_preprocessing.ipynb
3. notebooks/03_model_training_and_feature_selection.ipynb
```

Make sure the selected kernel is:

```text
Python (student_dropout_ml)
```

## Common Issues

### Streamlit uses the wrong Python environment

If Streamlit loads packages from the wrong environment, use:

```bash
conda activate student_dropout_ml
python -m streamlit run app/app.py
```

### Jupyter kernel does not appear

Reinstall the kernel:

```bash
conda activate student_dropout_ml
python -m ipykernel install --user --name student_dropout_ml --display-name "Python (student_dropout_ml)"
```

### Model file is missing

Make sure these files exist:

```text
models/final_mvp_model.pkl
models/model_metadata.json
models/mvp_features.json
```

If they are missing, rerun:

```text
notebooks/03_model_training_and_feature_selection.ipynb
```
