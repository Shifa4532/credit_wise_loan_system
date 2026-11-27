# Binary Classification Model for Credit Risk Assessment.

A machine learning project that predicts whether a loan application will be approved, using applicant finances, credit history, and personal details. Three classification models are compared, and the best one (Logistic Regression) reaches **87.5% accuracy** on held-out data.

## Problem
Banks review many loan applications and need a fast, consistent first-pass decision. This project builds a binary classifier on 1,000 past applications to predict `Loan_Approved` (Yes/No) and explores which factors matter most.

## Dataset
- **File:** `loan_approval_data.csv`
- **Size:** 1,000 applicants x 20 columns
- **Features:**
  - Financial: applicant income, coapplicant income, savings, collateral value, existing loans, debt-to-income (DTI) ratio, credit score
  - Loan details: loan amount, loan term, loan purpose, property area
  - Personal: age, gender, marital status, dependents, education level, employment status, employer category
- **Target:** `Loan_Approved` (Yes/No)

## Approach
1. **Data cleaning:** about 50 values were missing in every column. Categorical gaps were filled with the most frequent value and numeric gaps with the mean (`SimpleImputer`). Dropped the `Applicant_ID` column.
2. **Exploratory analysis:** compared distributions and boxplots of key features against the approval outcome.
3. **Encoding:**
   - `Education_Level` and `Loan_Approved` label encoded
   - Employment status, marital status, loan purpose, property area, gender, and employer category one-hot encoded (first category dropped)
4. **Train/test split:** 80/20 split (800 training, 200 test rows) with `random_state=42`, then standardized with `StandardScaler` (fit on the training set only).
5. **Baseline models:** trained Logistic Regression, K-Nearest Neighbors (k = 5), and Gaussian Naive Bayes.
6. **Feature engineering:** replaced `Credit_Score` and `DTI_Ratio` with their squared versions to capture non-linear effects, then retrained all three models.
7. **Evaluation:** accuracy, precision, recall, F1 score, and confusion matrix, with "approved" as the positive class.

## Exploratory findings
- **Credit score** is the clearest separator. No approvals appear below a score of roughly 650, and approved applicants have a noticeably higher median (about 725 vs. about 645).
- **DTI ratio** is lower for approved applicants (median about 0.25 vs. about 0.40).
- Approval rates rise with applicant income, but income alone separates the groups much less than credit score or DTI.
- Age and savings look similar for approved and rejected applicants.

## Results
Test set: 200 applicants (139 rejected, 61 approved). Predicting "rejected" for everyone would give 69.5% accuracy, so that is the baseline to beat.

| Model | Features | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Logistic Regression | Original | 0.865 | 0.783 | 0.770 | 0.777 |
| KNN (k = 5) | Original | 0.760 | 0.627 | 0.525 | 0.571 |
| Naive Bayes | Original | 0.865 | 0.804 | 0.737 | 0.769 |
| **Logistic Regression** | **Squared credit score and DTI** | **0.875** | **0.790** | **0.803** | **0.797** |
| KNN (k = 5) | Squared credit score and DTI | 0.755 | 0.620 | 0.508 | 0.559 |
| Naive Bayes | Squared credit score and DTI | 0.865 | 0.783 | 0.770 | 0.777 |

**Best model:** Logistic Regression with the engineered features. Its confusion matrix is `[[126, 13], [12, 49]]`: it correctly flagged 49 of 61 approvals and made 13 false approvals. KNN performed worst, likely because distance-based methods struggle with many one-hot encoded features.

**Note:** the test set is small and uses a single split, so the improvement from feature engineering (two more correct predictions) is suggestive rather than conclusive.

## Tech stack
Python, Jupyter Notebook, pandas, NumPy, matplotlib, seaborn, scikit-learn

## Author
Shifa · [GitHub](https://github.com/Shifa4532)
