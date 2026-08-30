# Loan Approval Prediction System

This project is a machine-learning capstone application for predicting whether a loan application is likely to be approved.

## Features

- Data loading and inspection
- Missing-value handling through an sklearn preprocessing pipeline
- Categorical encoding with one-hot encoding
- Four model candidates:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - Gradient Boosting
- Model comparison using accuracy, precision, recall, F1 score, and ROC-AUC
- Saved best model using Joblib
- Streamlit user interface
- Single applicant prediction
- Confidence score
- Feature importance visualization
- Loan approval analytics dashboard
- Batch CSV prediction
- Downloadable PDF prediction report

## Project Structure

```text
loan_approval_system/
├── app.py
├── train_model.py
├── reporting.py
├── requirements.txt
├── README.md
├── data/
│   └── loan_approval.csv
└── models/
    ├── loan_approval_model.joblib
    └── model_metrics.csv
```

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Train the model:

```bash
python train_model.py
```

Start the app:

```bash
streamlit run app.py
```

## Dataset

The model uses the instructor-provided `loan_approval.csv` dataset. The target column is `Loan_Status`, where `Y` means approved and `N` means not approved.

## Note

This is an educational prediction system. It should not be used as the only basis for real financial approval decisions.
  
