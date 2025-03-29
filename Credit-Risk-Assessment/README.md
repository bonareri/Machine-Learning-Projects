# Credit Risk Assessment: Predicting Loan Defaults

## 1. Introduction

This project aims to build a Credit Risk Assessment system to help financial institutions evaluate the risk of loan applicants. Effective credit risk assessment is crucial in minimizing loan defaults, improving lending decisions, and ensuring financial stability.

## 2. Problem Statement

Financial institutions face challenges in accurately assessing borrowers' creditworthiness. Poor risk evaluation can lead to high default rates, financial losses, and an unstable credit market. This project leverages machine learning techniques to predict credit risk, enhance decision-making, and minimize potential losses.

## 3. Objectives

This project aims to achieve the following goals:

- **Analyzing Historical Loan and Credit Data** – Identify patterns and trends in borrower behavior.
- **Implementing Machine Learning Models for Risk Prediction** – Develop and train models such as Logistic Regression, Random Forest, and XGBoost to classify loan applicants as low-risk or high-risk.
- **Evaluating Model Accuracy and Optimizing Performance** – Assess model performance using key metrics and fine-tune hyperparameters to improve prediction accuracy.

## 4. Data Collection

### Data Source

The dataset used for this project can be sourced from:

- **Public Credit Risk Datasets:**
  - Kaggle Lending Club loan data

### Features in the Dataset  

The dataset consists of **255,347** loan applications with **18 features**, categorized into different aspects of a borrower's financial profile.  

**1. Demographic Information**  
- **Age** – The borrower's age.  
- **Education** – Highest level of education attained.  
- **EmploymentType** – Type of employment (e.g., Salaried, Self-Employed).  
- **Income** – The borrower’s annual income.  
- **MaritalStatus** – Indicates if the borrower is Single, Married, etc.  
- **HasDependents** – Whether the borrower has dependents.  

**2. Credit History**  
- **CreditScore** – A numerical representation of the borrower’s creditworthiness.  
- **NumCreditLines** – Number of existing credit lines.  

**3. Loan Details**  
- **LoanAmount** – The total amount borrowed.  
- **InterestRate** – The percentage interest charged on the loan.  
- **LoanTerm** – The duration of the loan in months.  
- **LoanPurpose** – The purpose for which the loan was taken (e.g., home, education, business).  

**4. Financial Status & Risk Indicators**  
- **MonthsEmployed** – Number of months the borrower has been employed.  
- **DTIRatio** – Debt-to-Income Ratio (Total Debt / Income).  
- **HasMortgage** – Indicates if the borrower has an existing mortgage.  
- **HasCoSigner** – Whether the loan has a co-signer.  

**5. Loan Default Indicator**  
- **Default** (Target Variable) – A binary indicator (0 = No Default, 1 = Default).  


## 5. Exploratory Data Analysis

### Summary Statistics  

The table below provides summary statistics for the numerical features in the dataset:

| Feature          | Count      | Mean      | Std Dev   | Min   | 25%   | 50%   | 75%   | Max   |
|-----------------|-----------|----------|----------|-------|-------|-------|-------|-------|
| **Age**         | 255,347   | 43.50    | 14.99    | 18    | 31    | 43    | 56    | 69    |
| **Income**      | 255,347   | 82,499   | 38,963   | 15,000 | 48,825 | 82,466 | 116,219 | 149,999 |
| **LoanAmount**  | 255,347   | 127,578  | 70,841   | 5,000 | 66,156 | 127,556 | 188,985 | 249,999 |
| **CreditScore** | 255,347   | 574.26   | 158.90   | 300   | 437   | 574   | 712   | 849   |
| **MonthsEmployed** | 255,347 | 59.54  | 34.64    | 0     | 30    | 60    | 90    | 119   |
| **NumCreditLines** | 255,347 | 2.50   | 1.12     | 1     | 2     | 2     | 3     | 4     |
| **InterestRate** | 255,347  | 13.49   | 6.64     | 2.00  | 7.77  | 13.46  | 19.25  | 25.00  |
| **LoanTerm**    | 255,347   | 36.03   | 16.97    | 12    | 24    | 36    | 48    | 60    |
| **DTIRatio**    | 255,347   | 0.50    | 0.23     | 0.10  | 0.30  | 0.50  | 0.70  | 0.90  |
| **Default**     | 255,347   | 0.12    | 0.32     | 0     | 0     | 0     | 0     | 1     |

---

**Key Insights**  

- **Age & Income**: Borrowers are mostly middle-aged (**median: 43**), with income widely spread (**median: 82,499**, max: **149,999**).  
- **Loan Amount & Credit Score**: Median loan amount is **127,556**, with a moderate credit score (**median: 574**). **25% have scores below 437**, indicating higher risk.  
- **Loan Terms & Interest Rates**: Most loans are for **24-48 months**, with interest rates reaching **25%** for riskier applicants.  
- **Debt-to-Income Ratio (DTI)**: **Median DTI is 0.50**, meaning half of borrowers allocate **50% of income** to debt repayment. **DTI > 0.70** signals higher default risk.  

### Understanding Data Distribution:

**Imbalance in distribution:** 

![image](https://github.com/user-attachments/assets/b4f1235f-f831-4f8e-8c2b-2d1d22dabeed)

- Non-defaulters make up **88.4%** of the dataset, while defaulters account for only **11.6%**.
- This class imbalance may affect model performance, requiring techniques like oversampling, undersampling, or adjusting class weights.  

### KDE Plots of Numerical Features by Default Status

![image](https://github.com/user-attachments/assets/b91c7b75-abad-4a6a-aad1-2bf59ed914d1)

 **Insights from KDE Plots**

- **Younger borrowers (20-30)** are more likely to default, while non-defaulters are evenly spread across ages.  
- **Lower income** and **higher loan amounts** increase the likelihood of default.  
- **Defaulters have lower credit scores**, while non-defaulters generally have scores above 700.  
- **Shorter employment duration** is linked to higher default rates.  
- **Higher interest rates** and **higher Debt-to-Income (DTI) ratios** correlate with more defaults.  
- **Loan terms appear fixed at certain intervals**, with non-defaulters favoring shorter terms.  

### Correlation Analysis

![image](https://github.com/user-attachments/assets/efadf642-cefb-4091-9a50-5f7834ef1e6b)

**Key Takeaways from Correlation Analysis**

- **Age has the strongest negative correlation** (-0.168), meaning younger borrowers are more likely to default.  
- **Interest Rate has the strongest positive correlation** (0.131), meaning higher interest rates correspond to more defaults.  
- **Higher Loan Amounts** (0.087) slightly increase default risk due to a higher repayment burden.  
- **Longer Employment Duration** (-0.097) reduces default risk, suggesting that job stability helps prevent defaults.  
- **Higher Income** (-0.099) is associated with lower default risk, indicating that financially stable borrowers are less likely to default.  
- **Credit Score** (-0.034) shows a weak negative correlation, meaning higher scores slightly reduce default probability.  
- **Most correlations are weak to moderate**, implying that **default is influenced by multiple factors rather than a single one**.

![image](https://github.com/user-attachments/assets/dd571406-001d-440a-9f07-7a55f23088ef)

####  1️⃣ Key Observations  
- Defaulters (Default = 1) **tend to have lower credit scores** compared to non-defaulters.  
- The median credit score for defaulters is significantly lower than that of non-defaulters.  
- There is a wider spread of credit scores among non-defaulters, indicating **higher variability** in their creditworthiness.

#### 2️⃣ Business Implications  
- **Credit Score as a Risk Indicator:**  
  - A **lower credit score increases the likelihood of default**.  
  - Lending institutions should **prioritize credit score thresholds** when assessing loan applications.   

![image](https://github.com/user-attachments/assets/bb92072b-c867-404a-909f-f135a43fedf3)

#### 🔍 Observations  
- **Defaulters (Red) cluster in the top-left**, indicating higher defaults among **low-income, high-loan** borrowers.  
- **Non-defaulters (Green) are more evenly spread**, with many having **higher incomes and lower loan amounts**.

#### 💡 Business Implications  
- **Higher risk** for low-income borrowers requesting large loans.  
- **Stricter eligibility criteria & income-based loan caps** can help mitigate default risk.

![image](https://github.com/user-attachments/assets/da5732c6-e778-40a3-8b16-7d7abbc04317)

**Insights**
- Higher DTI ratios correlate with increased default risk.
- Borrowers with a DTI ratio > 0.7 are more likely to default.

![image](https://github.com/user-attachments/assets/7b9b1f7e-bc02-431e-8f07-7179381e4a70)

**Insights**
- Unemployed borrowers have a higher default rate than salaried employees. - Full-time employees are at a lower risk of default.

### Data Preprocessing

- **Class Imbalance Handling:**  
  - I addressed class imbalance using **SMOTE** to oversample the minority class and experimented with **class weighting** in model training.  

- **Feature Engineering:**  
  - Categorical variables were encoded using **One-Hot Encoding (OHE)** and **Label Encoding** where appropriate.  
  - Numerical features were scaled using **StandardScaler** to ensure consistency across different models.
  - Missing values were handled by dropping the rows with missing values, since the percentage of missing data was negligible. 

## 6. Model Development  

- **Baseline Models:**  
  - I started with **Logistic Regression** to establish a simple benchmark. 

- **Advanced Models:**  
  - **Random Forest** helped improve predictive performance through ensemble learning.  
  - **XGBoost** was implemented to leverage boosting for higher accuracy and recall.  

- **Hyperparameter Tuning:**  
  - I used **RandomizedSearchCV** and **GridSearchCV** to fine-tune key hyperparameters for each model.  

- **Model Evaluation:**  
  - I compared models based on **accuracy, precision, recall, F1-score, and AUC-ROC**.  
  - Since class imbalance was a concern, I paid close attention to **recall and F1-score** to ensure the minority class was well predicted.
 
## 7. Model Evaluation Summary  

### Baseline Model: Logistic Regression 
**Classification Report Analysis**

| Metric      | Class 0 (No Default) | Class 1 (Default) | Macro Avg | Weighted Avg |
|-------------|----------------------|-------------------|-----------|--------------|
| Precision   | 0.93                 | 0.22              | 0.57      | 0.84         |
| Recall      | 0.74                 | 0.56              | 0.65      | 0.72         |
| F1-score    | 0.82                 | 0.32              | 0.57      | 0.76         |

<table>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/b10876ce-741d-4d9b-ba3c-cbf8d5dd66f9" width="500"></td>
    <td><img src="https://github.com/user-attachments/assets/684e4d08-eedf-450e-9b79-3a97b6be7f61" width="500"></td>
  </tr>
</table>

- **Key Issues:**  
  - Logistic Regression is not performing well for Class 1.
  - A recall of 56% for Class 1 means nearly half of the positive cases are missed.
  - Class imbalance led to poor recall for defaulters.  
  - Majority class (non-defaulters) was predicted well, but many defaulters were missed.  

### Random Forest
**Classification Report Analysis**

| Metric      | Class 0 (No Default) | Class 1 (Default) | Macro Avg | Weighted Avg |
|-------------|----------------------|-------------------|-----------|--------------|
| Precision   | 0.94                 | 0.22              | 0.58      | 0.85         |
| Recall      | 0.71                 | 0.63              | 0.67      | 0.70         |
| F1-score    | 0.81                 | 0.33              | 0.57      | 0.75         |

<table>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/a2c20d03-60d4-424c-b820-28034900e9ce" width="500"></td>
    <td><img src="https://github.com/user-attachments/assets/10441f0e-303b-46e9-a424-fea16efed7d8" width="500"></td>
  </tr>
</table>

### XGBoost 
**Classification Report Analysis**

| Metric      | Class 0 (No Default) | Class 1 (Default) | Macro Avg | Weighted Avg |
|-------------|----------------------|-------------------|-----------|--------------|
| Precision   | 0.94                 | 0.22              | 0.58      | 0.85         |
| Recall      | 0.70                 | 0.64              | 0.67      | 0.69         |
| F1-score    | 0.80                 | 0.33              | 0.56      | 0.75         |

<table>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/b063bfba-eb4c-4d97-ba29-d0eca5d1aa8e" width="500"></td>
    <td><img src="https://github.com/user-attachments/assets/2232760e-ee16-4726-a083-3c5c8059cbec" width="500"></td>
  </tr>
</table>

### XGBoost Tuned
**Classification Report Analysis**

| Metric      | Class 0 (No Default) | Class 1 (Default) | Macro Avg | Weighted Avg |
|-------------|----------------------|-------------------|-----------|--------------|
| Precision   | 0.91                 | 0.34              | 0.62      | 0.84         |
| Recall      | 0.93                 | 0.29              | 0.61      | 0.85         |
| F1-score    | 0.92                 | 0.31              | 0.61      | 0.85         |

<table>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/782db019-133e-40ff-9359-3dcc71463af4" width="500"></td>
    <td><img src="https://github.com/user-attachments/assets/d9107c60-bb11-41e6-82ae-08838f6065c9" width="500"></td>
  </tr>
</table>

---

### Summary Performance 

| Model        | Accuracy | Precision (Defaulters) | Recall (Defaulters) | F1-score (Defaulters) |
|-------------|----------|-----------------------|---------------------|----------------------|
| **Logistic Regression** | 72% | 0.22 | 0.56 | 0.32 |
| **Random Forest** | 70% | 0.22 | 0.63 | 0.33 |
| **XGBoost** | 69% | 0.22 | 0.64 | 0.33 |
| **XGBoost Tuned** | 85% | 0.34 | 0.29 | 0.31 |

### Summary of Observations  

#### Accuracy:  
- **XGBoost Tuned** has the highest accuracy at **85%**, significantly outperforming the other models (which range between **69%-72%**).  
- However, since the **dataset is imbalanced**, accuracy alone is not a reliable metric for defaulter detection.  

#### Precision (Defaulters):  
- Precision measures how many of the predicted defaulters are actually defaulters.  
- All models (except **XGBoost Tuned**) have a **low precision of 0.22**, meaning many false positives.  
- **XGBoost Tuned improves precision to 0.34**, which is better at reducing false alarms but may miss actual defaulters.  

#### Recall (Defaulters):  
- Recall is crucial for defaulter detection as it shows how many actual defaulters are correctly identified.  
- **XGBoost (0.64) and Random Forest (0.63) have the highest recall**, meaning they detect more defaulters but at the cost of more false positives.  
- **XGBoost Tuned has the lowest recall (0.29)**, meaning it misses many actual defaulters.  

#### F1-Score (Defaulters):  
- **F1-score balances precision and recall.**  
- **Logistic Regression (0.32), Random Forest (0.33), and XGBoost (0.33)** have similar F1-scores.  
- **XGBoost Tuned (0.31) has a slightly lower F1-score**, indicating an imbalanced trade-off between precision and recall.  

#### Best Model for Detecting Defaulters:  
- **XGBoost is the best choice** due to its **highest recall (0.64)**, ensuring more defaulters are correctly identified.  
- **Random Forest is a close alternative**, with **recall (0.63)** and a similar F1-score (0.33).  
- **XGBoost Tuned sacrifices recall (0.29) for better precision (0.34)** but misses too many defaulters, making it less suitable for this objective.  

#### Conclusion:  
Since the goal is to **detect the highest number of defaulters**, **XGBoost is the best model** due to its **high recall (0.64)**, ensuring more defaulters are correctly identified.  

---
## 7. Future Improvements

- Implementing deep learning models for enhanced prediction accuracy.
- Exploring additional data sources for better risk assessment.
- Deploying the model as a web-based or API service for real-time credit scoring.
