# Team Task Division

Use this file to show the mentor that all four members have clear responsibilities in the Loan Approval Prediction System.

## Member 1 - Data Preparation and Model Training

Main files:
- `data/loan_approval.csv`
- `train_model.py`
- `models/loan_approval_model.joblib`
- `models/model_metrics.csv`

Responsibilities:
- Inspect the dataset.
- Check missing values and duplicates.
- Separate features from the target column `Loan_Status`.
- Handle missing values using the preprocessing pipeline.
- Encode categorical fields.
- Train the machine-learning models.
- Compare model performance using accuracy, precision, recall, F1 score, and ROC-AUC.
- Save the best model.

## Member 2 - Prediction Interface

Main file:
- `app.py`

Responsibilities:
- Build the Streamlit app layout.
- Create the single applicant prediction form.
- Add inputs for applicant information such as income, loan amount, credit history, education, and property area.
- Load the saved trained model.
- Display the prediction result: Approved or Not Approved.
- Display the model confidence score.

## Member 3 - Dashboard and Batch Prediction

Main file:
- `app.py`

Responsibilities:
- Build the analytics dashboard.
- Show total applications, approval rate, median loan amount, and missing values.
- Add charts for approval outcome count.
- Add charts for approval rate by credit history, property area, and education.
- Add feature importance visualization.
- Add batch CSV upload prediction.
- Add downloadable CSV results for batch predictions.

## Member 4 - PDF Report and Documentation

Main files:
- `reporting.py`
- `README.md`
- `requirements.txt`

Responsibilities:
- Generate a downloadable PDF report for a single prediction.
- Include applicant details, prediction result, and confidence score in the report.
- Document the project features.
- Explain how to install dependencies.
- Explain how to run the Streamlit app.
- Prepare the project for GitHub submission.

## Files to Upload to GitHub

Upload these files and folders:

```text
app.py
train_model.py
reporting.py
requirements.txt
README.md
TEAM_TASKS.md
.gitignore
data/
models/
```

Do not upload these generated folders:

```text
__pycache__/
.venv/
.agents/
.claude/
```
