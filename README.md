Demo Video :- https://nagarro-my.sharepoint.com/:v:/p/mohanpreet_singh/IQCpmwrf-uFZRYU973Wx_OqTAUrr9VCkKQRKkD7tBhOhdlY?e=YzD0oQ

Github Repo link :- https://github.com/mohanpreet-nagarro/nagp-ds-assignment.git

# Telco Customer Churn Prediction

## 1. Business Problem

A telecommunications company wants to identify customers who are likely to leave (churn) so the retention team can proactively contact them.

This project builds an end-to-end machine learning solution using the supplied IBM Telco Customer Churn dataset.

**Target:** `Churn` (`Yes` / `No`)

## 2. Project Structure

```text
customer_churn_project/
├── data/
│   ├── TelcoCustomerChurn.csv
│   └── TelcoCustomerChurn - Data Dictionary.csv
├── notebook/
│   └── churn_analysis.ipynb
├── model/
│   └── churn_pipeline.pkl
├── app.py
├── model_utils.py
├── train.py
├── requirements.txt
├── README.md
└── sample_request.json
```

`model_utils.py` defines the custom `FeatureEngineer` transformer used inside the saved pipeline. It must stay at the project root — both `train.py` (to save the model) and `app.py` (to load it) need to import it from the same location, or `joblib.load` will fail.

## 3. Dataset Understanding

The supplied dataset contains 7,043 customer records and 21 columns, including the target `Churn`.

`customerID` and the CSV `index` column are identifiers and are not used as model features.

`TotalCharges` is supplied as text, so the project converts it to numeric during preprocessing.

The data dictionary documents the meaning and expected values of the customer attributes.

## 4. Machine Learning Workflow

```text
Business Problem
      ↓
Load Dataset
      ↓
Data Understanding
      ↓
Cleaning / Missing Value Analysis
      ↓
EDA
      ↓
Feature Engineering
      ↓
70:30 Train/Test Split
      ↓
Preprocessing Pipeline
      ↓
Decision Tree Models
      ↓
Evaluation
      ↓
Feature Importance / Tree Interpretation
      ↓
Save Complete Pipeline
      ↓
FastAPI /predict
```

## 5. Feature Engineering

Two row-level features are created:

1. **AverageMonthlySpend**
   - `TotalCharges / tenure`
   - Gives an estimate of the customer's average historical monthly spend.
   - For customers with `tenure = 0` (brand-new signups with no billing history), this defaults to `0` rather than dividing by zero.

2. **ServiceCount**
   - Counts subscribed services marked `Yes`.
   - Represents the breadth of the customer's service relationship with the company.

These features are created inside the sklearn pipeline (`model_utils.FeatureEngineer`), so the same transformations are automatically applied to unseen API customers, including brand-new customers with zero tenure.

## 6. Data Leakage Prevention

The project avoids target leakage by:

- Removing `Churn` before model training.
- Removing `customerID` and the CSV `index` identifier.
- Fitting imputers and one-hot encoding only through the training pipeline.
- Using `OneHotEncoder(handle_unknown="ignore")` for unseen categorical values.
- Saving preprocessing and the model together as one pipeline.

## 7. Train/Test Split

The assignment requires:

- Training set: 70%
- Test set: 30%
- `random_state=42`

The implementation also uses `stratify=y` so the churn/non-churn ratio remains similar in both sets.

## 8. Decision Tree Models

Two configurations are compared.

### Model 1 - Baseline

- `max_depth=5`
- `min_samples_split=10`

### Model 2 - Recall Focused (selected as final)

- `max_depth=6`
- `min_samples_split=10`
- `min_samples_leaf=3`
- `class_weight="balanced"`

The second configuration gives more importance to the minority churn class.

For this business problem, recall is important because a false negative means a customer who was likely to churn was missed by the retention team.

### Model comparison (held-out test set, 30% split, `random_state=42`)

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Baseline | 0.794 | 0.614 | 0.601 | 0.607 |
| Recall Focused (final) | 0.746 | 0.515 | 0.758 | 0.613 |

The recall-focused model trades roughly 5 points of accuracy and 10 points of precision for a ~16-point gain in recall — a deliberate trade-off for a first-stage retention screening tool, where missing an actual churner is costlier than flagging a customer who wouldn't have churned.

## 9. Evaluation Metrics

The notebook calculates:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

### Confusion matrix (final model)

```
[[1151  401]
 [ 136  425]]
```

Out of 561 actual churners in the test set, the model correctly flags 425 (75.8% recall), at the cost of 401 false positives.

### Business interpretation

For churn prevention, **Recall should generally receive higher priority than Precision**.

High recall means the company catches more of the customers who are actually going to churn. A retention team can then decide which of those predicted customers are worth contacting.

However, precision should still be monitored because contacting too many customers who would not churn increases retention campaign cost.

## 10. How to Run

### Step 1 - Create and activate a virtual environment

Windows:

```powershell
python -m venv churn_env
churn_env\Scripts\Activate.ps1
```

If PowerShell blocks script execution, either use Command Prompt instead (`churn_env\Scripts\activate.bat`) or run once: `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`.

### Step 2 - Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 3 - Run the notebook

Open:

```text
notebook/churn_analysis.ipynb
```

in VS Code.

Select the `churn_env` Python interpreter/kernel and run all cells from top to bottom (Restart & Run All is recommended before submission, to confirm every cell runs cleanly in sequence).

The notebook performs the complete analysis, EDA, feature engineering, model training, evaluation, interpretation and model saving.

### Step 4 - Train/save the model from Python

From the project root:

```powershell
python train.py
```

This creates:

```text
model/churn_pipeline.pkl
```

and prints the model comparison table shown in section 8.

### Step 5 - Start the API

```powershell
uvicorn app:app --reload
```

Open Swagger UI:

```text
http://127.0.0.1:8000/docs
```

Use `POST /predict`.

## 11. Sample API Request

The file `sample_request.json` contains a valid customer payload.

Example:

```json
{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "No",
  "Dependents": "No",
  "tenure": 2,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "No",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "No",
  "StreamingMovies": "No",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 70.7,
  "TotalCharges": 151.65
}
```

**Verified response** (tested end-to-end against the trained pipeline):

```json
{
  "prediction": "Yes",
  "churn_probability": 0.7997
}
```

## 12. Invalid Input Handling

FastAPI/Pydantic validates:

- required fields
- numeric fields
- binary values such as `SeniorCitizen`
- supported categorical values
- non-negative tenure/charge values

Invalid requests receive an HTTP 422 validation response, before the request ever reaches the model.

## 13. Notebook Sections

The notebook is organized to match the assignment:

1. Business problem and objective
2. Import libraries
3. Load dataset and data dictionary
4. Data understanding
5. Data types and structure
6. Missing-value analysis
7. Duplicate analysis
8. Numerical/categorical feature identification
9. Target analysis
10. Data cleaning
11. EDA with more than five visualizations
12. Feature engineering
13. Train/test split
14. Leakage-safe preprocessing
15. Two Decision Tree configurations
16. Model comparison
17. Accuracy, Precision, Recall, F1
18. Confusion matrix
19. Feature importance
20. Decision Tree visualization
21. Final model selection
22. Save complete pipeline
23. Sample prediction

## 14. Key Findings to Discuss

The notebook calculates the actual values rather than hard-coding them. The business discussion should focus on patterns such as:

- Month-to-month customers are generally more vulnerable to churn.
- Short-tenure customers are important retention targets.
- Higher monthly charges can be associated with increased churn risk.
- Payment method and service configuration can provide useful churn signals.
- Longer contracts generally indicate stronger customer commitment.
- Recall is especially important when the business goal is to identify as many potential churners as possible.

These are observations to validate with the charts and model results; they should not be treated as universal rules outside this dataset.

## 15. Deliverables

This project provides:

- Complete Jupyter Notebook
- Python training script (`train.py`)
- Custom feature engineering module (`model_utils.py`)
- FastAPI REST API (`app.py`)
- Saved preprocessing + model pipeline (`model/churn_pipeline.pkl`)
- Requirements file
- README
- Sample API request (`sample_request.json`)
- Original dataset and data dictionary

## 16. Known Limitations

- The Decision Tree is trained on a historical snapshot; churn drivers can shift over time (pricing changes, competitor offers), so periodic retraining is recommended.
- `AverageMonthlySpend` defaults to `0` for customers with zero tenure — a reasonable placeholder for brand-new signups, but it means the model relies on other features (contract type, service selections) to assess risk for that segment.
- Precision (51.5%) means roughly half of customers flagged as "at risk" will not actually churn; this is an intentional trade-off given the business priority on recall, but campaign cost/capacity planning should account for it.