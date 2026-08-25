# Intelligent Loan Approval System

An end-to-end machine learning project for predicting whether a loan application will be approved based on applicant financial, demographic, employment, and loan-related information.

## Project Overview

This project builds and compares multiple classification models for loan approval prediction. The workflow covers:

- Data loading and initial inspection
- Missing-value analysis and handling
- Exploratory Data Analysis (EDA)
- Target encoding
- Train/test splitting
- Categorical feature encoding
- Correlation analysis
- Feature engineering
- Feature scaling
- Model training and evaluation
- Confusion-matrix visualization
- Comparison using Accuracy, Precision, Recall, F1-score, and ROC-AUC

## Dataset

The notebook loads the dataset from:

```text
loan_approval_data.csv
```

The dataset contains **1,000 rows and 20 columns** before preprocessing. Each column initially contains 950 non-null values, resulting in 50 missing values per column.

### Features

| Feature | Description |
|---|---|
| `Applicant_ID` | Applicant identifier; removed before modeling |
| `Applicant_Income` | Applicant income |
| `Coapplicant_Income` | Coapplicant income |
| `Employment_Status` | Applicant employment status |
| `Age` | Applicant age |
| `Marital_Status` | Applicant marital status |
| `Dependents` | Number of dependents |
| `Credit_Score` | Applicant credit score |
| `Existing_Loans` | Number of existing loans |
| `DTI_Ratio` | Debt-to-income ratio |
| `Savings` | Applicant savings |
| `Collateral_Value` | Value of collateral |
| `Loan_Amount` | Requested loan amount |
| `Loan_Term` | Loan term |
| `Loan_Purpose` | Purpose of the loan |
| `Property_Area` | Property area/category |
| `Education_Level` | Applicant education level |
| `Gender` | Applicant gender |
| `Employer_Category` | Employer category |
| `Loan_Approved` | Target variable: `No` or `Yes` |

## Data Preprocessing

### 1. Remove unnecessary column

`Applicant_ID` is removed because it is an identifier and is not useful as a predictive feature.

### 2. Handle missing target values

Rows where `Loan_Approved` is missing are removed.

After this step, the dataset contains **950 usable records**.

### 3. Target encoding

The target is converted from text to binary values:

```python
df["Loan_Approved"] = df["Loan_Approved"].map({
    "No": 0,
    "Yes": 1
}).astype(int)
```

- `No` → `0`
- `Yes` → `1`

### 4. Train/test split

The data is split into:

- **80% training data**
- **20% testing data**
- `random_state=42`
- `stratify=y`

The stratified split preserves the class distribution between the training and testing sets.

### 5. Missing-value imputation

Numeric features use the **mean** strategy:

```python
SimpleImputer(strategy="mean")
```

Categorical features use the **most frequent** strategy:

```python
SimpleImputer(strategy="most_frequent")
```

The imputers are fitted on the training data and then applied to the test data to avoid data leakage.

### 6. Feature encoding

`Education_Level` is ordinal encoded:

```text
Not Graduate → 0
Graduate     → 1
```

The following categorical features are one-hot encoded:

- `Employment_Status`
- `Marital_Status`
- `Loan_Purpose`
- `Property_Area`
- `Gender`
- `Employer_Category`

The encoder uses:

```python
OneHotEncoder(
    drop="first",
    sparse_output=False,
    handle_unknown="ignore",
    dtype=int
)
```

### 7. Feature engineering

Two squared features are created:

```python
Credit_Score_sq = Credit_Score ** 2
DTI_Ratio_sq = DTI_Ratio ** 2
```

These features allow the models to capture possible nonlinear relationships.

### 8. Feature scaling

`StandardScaler` is used to standardize the features:

```python
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

The scaler is fitted only on the training data and then used to transform the test data.

## Exploratory Data Analysis

The notebook performs several EDA steps, including:

- Loan approval class distribution
- Categorical feature analysis
- Applicant and coapplicant income distributions
- Outlier analysis using box plots
- Credit score distribution by loan approval
- Applicant income distribution by loan approval
- Correlation heatmap
- Correlation of features with the target

The target distribution after removing missing target values is:

| Loan Approval | Count |
|---|---:|
| No | 652 |
| Yes | 298 |
| **Total** | **950** |

This shows that the dataset is not perfectly balanced, so evaluation should consider metrics beyond accuracy.

## Machine Learning Models

Three classification approaches are trained and evaluated:

### 1. Logistic Regression

```python
LogisticRegression(
    max_iter=1000,
    random_state=42
)
```

### 2. Gaussian Naive Bayes

```python
GaussianNB()
```

### 3. K-Nearest Neighbors

KNN is tuned using `GridSearchCV` with:

```python
n_neighbors = [3, 5, 7, 9]
```

Five-fold cross-validation is used, with **precision** as the grid-search scoring metric.

## Model Evaluation

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrix
- Classification report

### Results

The notebook produced the following test-set results:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.895 | 0.845 | 0.817 | 0.831 | 0.941 |
| Gaussian Naive Bayes | 0.889 | 0.810 | 0.850 | 0.829 | 0.952 |
| K-Nearest Neighbors | 0.853 | 0.820 | 0.683 | 0.745 | 0.900 |

### Classification Reports

#### Logistic Regression

- Accuracy: **89%**
- Precision for approved loans: **0.84**
- Recall for approved loans: **0.82**
- F1-score for approved loans: **0.83**

#### Gaussian Naive Bayes

- Accuracy: **89%**
- Precision for approved loans: **0.81**
- Recall for approved loans: **0.85**
- F1-score for approved loans: **0.83**

#### K-Nearest Neighbors

- Accuracy: **85%**
- Precision for approved loans: **0.82**
- Recall for approved loans: **0.68**
- F1-score for approved loans: **0.75**

## Model Comparison

Based on the notebook's test results:

- **Logistic Regression** achieved the highest accuracy (**0.895**).
- **Gaussian Naive Bayes** achieved the highest ROC-AUC (**0.952**) and the highest recall for approved loans (**0.850**).
- **K-Nearest Neighbors** produced the lowest overall performance among the three models.

The best model depends on the business objective. If identifying more genuinely approved applications is especially important, recall may be more relevant; if overall ranking/discrimination is important, ROC-AUC is useful.

## Project Workflow

```text
Raw Loan Data
      │
      ▼
Initial Data Inspection
      │
      ▼
Remove Applicant_ID
      │
      ▼
Remove Rows with Missing Target
      │
      ▼
Exploratory Data Analysis
      │
      ▼
Encode Target
      │
      ▼
Train/Test Split (80/20)
      │
      ▼
Missing-Value Imputation
      │
      ▼
Categorical Encoding
      │
      ▼
Correlation Analysis
      │
      ▼
Feature Engineering
      │
      ▼
Feature Scaling
      │
      ▼
Train Models
      │
      ├── Logistic Regression
      ├── Gaussian Naive Bayes
      └── K-Nearest Neighbors + GridSearchCV
      │
      ▼
Evaluate Models
      │
      ▼
Compare Performance
```

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

## Python Libraries

```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score
)
```

## How to Run

### 1. Clone or download the project

Make sure the notebook and dataset are in the appropriate project directory.

### 2. Install dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

### 3. Start Jupyter Notebook

```bash
jupyter notebook
```

### 4. Open the notebook

Open:

```text
Intelligent_Loan_Approval_System(3).ipynb
```

### 5. Add the dataset

Make sure:

```text
loan_approval_data.csv
```

is available at the path expected by the notebook.

### 6. Run the notebook

Run the cells from top to bottom to reproduce the preprocessing, EDA, model training, and evaluation results.

## Project Structure

```text
Intelligent-Loan-Approval-System/
│
├── Intelligent_Loan_Approval_System(3).ipynb
├── loan_approval_data.csv
└── README.md
```

## Key Takeaways

1. The project demonstrates a complete supervised machine-learning workflow for binary loan approval classification.
2. The preprocessing pipeline handles missing values, categorical variables, feature scaling, and feature engineering.
3. Three classification models are compared using several evaluation metrics.
4. Logistic Regression gives the highest test accuracy.
5. Gaussian Naive Bayes gives the highest ROC-AUC and recall for the approved-loan class.
6. KNN performs below the other two models on the reported test metrics.

## Future Improvements

Possible extensions include:

- Hyperparameter tuning for Logistic Regression and Gaussian Naive Bayes
- More extensive KNN tuning
- ROC curves for all models
- Precision-recall curves
- Cross-validation comparison across all models
- Feature importance or model interpretability
- Threshold optimization based on lending business costs
- Saving the best trained model with `joblib` or `pickle`
- Building a web interface or dashboard for interactive loan prediction
- Adding a prediction API for deployment

## Author

**Mehedi Hasan**

---

*This README is based on the workflow, code, and results contained in the provided Jupyter Notebook.*
