# Cryptocurrecncy Price Prediction

## 1. Introduction

This project aims to build a Cryptocurrency Price Prediction system to help traders and investors make data-driven decisions. Cryptocurrency markets are highly volatile, and predicting price movements can minimize risks, maximize profits, and enhance trading strategies.

## 2. Problem Statement 

Cryptocurrency prices are highly volatile, making it difficult for traders and investors to make informed decisions. This volatility leads to financial losses, market manipulation, and investment uncertainty. The project aims to address these challenges by predicting future prices using machine learning and statistical analysis, providing data-driven insights, and reducing uncertainty in cryptocurrency trading.

## 3. Objectives

This project aims to achieve the following goals:

- **Analyzing Historical Price Trends** – Study past price trends to understand market behavior and identify patterns.
- **Implementing Machine Learning and Time series Models for Prediction** – Develop and train models like LSTMs, Random Forest, and ARIMA to predict future cryptocurrency prices.
- **Evaluating Model Accuracy and Improving Predictions** – Assess the performance of different models and fine-tune them to improve prediction accuracy.

## 4. Data Collection

### Data Source

The data for this project is sourced using the following APIs:

- **Price Data:**  
  - **Source:** Yahoo Finance API via the `yfinance` library  
  - **Data:** Open, high, low, close, and volume metrics.

- **Circulating Supply:**
 - **Source:** CoinGecko API
 - **Data:** The circulating supply is used to calculate the Market Cap of each coin, by multiplying the Close price by the circulating supply during that time period.

### Features in the Dataset

The dataset includes the following key features:

- **Open** – The price at which the cryptocurrency opened during a specific time period.
- **Close** – The price at which the cryptocurrency closed during the specific time period.
- **High** – The highest price during the time period.
- **Low** – The lowest price during the time period.
- **Volume** – The total number of units traded during the time period.
- **Market Cap** - The total market value of the cryptocurrency during the time period.
- **Exponential Moving Averages (EMA):**
  - **EMA_50** – Short-term trend indicator (50-day EMA).
  - **EMA_200** – Long-term trend indicator (200-day EMA).
- **Daily Return:** Measures the percentage change in closing prices between consecutive days.
- **Relative Strength Index (RSI):** Identifies overbought (>70) or oversold (<30) conditions.

## 5. Exploratory Data Analysis

### Cryptocurrency Price Statistics 

| Coin | Count | Mean       | Std        | Min      | 25%       | 50%       | 75%       | Max       |
|------|-------|------------|------------|----------|-----------|-----------|-----------|-----------|
| BTC  | 3807  | 20588.04   | 23711.62   | 178.10   | 1278.03   | 9508.99   | 32941.57  | 106146.27 |
| ETH  | 2658  | 1520.86    | 1235.34    | 84.31    | 274.32    | 1455.79   | 2472.06   | 4812.09   |
| SOL  | 1775  | 72.48      | 71.12      | 0.52     | 18.25     | 35.56     | 136.75    | 261.87    |

**Summary:**

- The **mean closing price** for **BTC** is **$20,588**, with a **high standard deviation** of **$23,711**, indicating significant volatility.
- **BTC** prices range from a **minimum of $178** to a **maximum of $106,146**, showing a large price spread.
- The **median closing price** for **BTC** is **$9,508**, much lower than the **mean**, indicating a **right-skewed distribution**.
  
- For **ETH**, the **mean price** is **$1,520**, and its price range is from **$84** to **$4,812**, with a **standard deviation** of **$1,235**.
- The **median closing price** for **ETH** is **$1,455**, suggesting a mild **right-skewed distribution**.

- **SOL** has a **mean closing price** of **$72.48**, and a **standard deviation** of **$71.12**, indicating high volatility.
- The **price range for SOL** spans from **$0.52** to **$261.87**, showing considerable fluctuations, though smaller than **BTC** and **ETH**.
- The **median closing price** for **SOL** is **$35.56**, reinforcing a **right-skewed distribution**.

These observations highlight the volatility and price distributions for **BTC**, **ETH**, and **SOL**, with **BTC** showing the widest price range and highest volatility.
 
### Frequency Distribution

![image](https://github.com/user-attachments/assets/1a12dba9-b3a9-43bc-b7a4-85276e753dce)

- The close price shows a **right-skewed distribution**, indicating that lower prices were more frequent.
- There are **multiple peaks**, suggesting different historical price levels or market phases.  

### Correlation Martix

![image](https://github.com/user-attachments/assets/2c4751e0-4b73-4b84-a3ac-862efc82e0df)

## 6. Data Analysis

### Closing Price Over Time

![image](https://github.com/user-attachments/assets/93fb0b22-489e-40ab-bb68-aaca3e100d6f)

![image](https://github.com/user-attachments/assets/567309a6-695a-461b-ac30-ce357237217b)


**1️⃣ Bitcoin (BTC) Dominates Price Trends**
- **BTC (orange)** remains the highest-priced cryptocurrency.
- Surged past **$100,000 in 2024**, showing strong market confidence.
- **Historical peaks in 2017, 2021, and 2024** indicate repeated bull cycles.

**2️⃣ Ethereum (ETH) Shows Moderate Growth**
- **ETH (blue)** has a much lower price range compared to BTC.
- Peaked around **$5,000** in previous bull cycles.
- **Gradual upward trend**, showing solid adoption.

**3️⃣ Solana (SOL) Remains Relatively Lower in Price**
- **SOL (green)** shows price spikes after 2021 but remains below BTC & ETH.

### Market Capitalization

![image](https://github.com/user-attachments/assets/f8492bd6-fff0-4b25-9de9-a6b1bde9f2ae)


**1️⃣ Bitcoin (BTC) Leads the Market**
- **BTC (orange)** has the highest market capitalization, peaking above **$2 trillion**.
- Significant **growth during bull runs** (2017, 2021, 2024).
- **Recent 2024 surge** suggests renewed investor confidence.

**2️⃣ Ethereum (ETH) Shows Strong Growth**
- **ETH (blue)** follows BTC but at a lower magnitude.
- Peaked around **$500 billion** in 2021 but remains **steadily increasing**.
- Indicates **strong network utility and adoption**.

**3️⃣ Solana (SOL) Gains Traction**
- **SOL (green)** had a late start but saw significant growth post-2021.
- **Smaller market cap** compared to BTC & ETH but shows steady **adoption and resilience**.

**4️⃣ Market Cycles Are Clearly Visible**
- **Boom and bust cycles** are evident (2021 bull run, 2022 bear market).
- **Post-2023 recovery** shows renewed market interest.

### Trading Volume Analysis

![image](https://github.com/user-attachments/assets/571be646-4d9c-463d-87f6-07b6371a8ef0)


1️⃣ Bitcoin (BTC) Dominates Trading Volume  
- **BTC (orange)** has the highest trading volume over time, especially during market peaks.  
- Major **spikes align with market cycles** (e.g., 2021 bull run).  

2️⃣ Ethereum (ETH) Has Consistently High Volume  
- **ETH (blue)** follows BTC’s trend but at a lower scale.  
- Shows **sustained liquidity**, indicating strong investor interest.  

3️⃣ Solana (SOL) Gained Traction Post-2020  
- **SOL (green)** had minimal trading before 2020 but grew rapidly.  
- Lower volume than BTC & ETH, but **trading activity is increasing**.  

 4️⃣ Volume Spikes Correlate with Market Events  
- **2021:** Crypto bull run → **Highest trading activity ever recorded**.  
- **2022:** Market crash → **Sharp spikes, indicating panic selling**.  
- **2024:** Volume stabilizes but remains **volatile, especially for BTC**.

### Moving Averages (EMA)

![image](https://github.com/user-attachments/assets/5f68a070-e34c-4bde-86b3-1a9ac4580a82)

### Volatility Analysis Using Rolling Standard Deviation

Volatility measures how much prices fluctuate over time. Higher volatility indicates higher risk but also higher potential returns.

![image](https://github.com/user-attachments/assets/f5caeb7d-c419-4f67-aeea-e4df04ac0d12)


 1️⃣ Bitcoin (BTC) Shows the Most Stability  
- BTC (orange) maintains relatively low volatility over time.  
- This suggests it is more **established** and less reactive to short-term market movements.  

 2️⃣ Ethereum (ETH) and Solana (SOL) Are More Volatile  
- ETH (blue) experiences **moderate fluctuations**, especially during market shifts.  
- SOL (purple) has the **highest volatility**, with frequent sharp spikes.  
- Post-2021, **SOL's volatility exceeds ETH & BTC**, indicating **higher speculative activity**.  

3️⃣ Volatility Spikes Align with Major Market Events  
- 2018: Crypto market crash → Sudden surge in volatility.  
- 2020: Pandemic-driven uncertainty → Increased market swings.  
- 2021: **Bull run & corrections** → Highest volatility levels observed.

## 7. Data Preprocessing

To prepare the dataset for analysis and model training, the following preprocessing steps were performed:

- **Feature Engineering:** I created additional features such as Exponential Moving Averages (EMA_50, EMA_200) and Daily Return to gain deeper insights into the price trends.
- **Feature Scalling:** Since the 'Close' prices was not normally distributed, I applied the **Min-Max Scaler** to normalize the data. This transformed the values into a range between 0 and 1, ensuring the model could better learn from the data.
- **Data Splitting**
  - **Chronological Order:**  
    Data is sorted by date to preserve temporal relationships.

  - **80/20 Split:**  
    - **Training Set (80%):** Earliest data for model learning.  
    - **Test Set (20%):** Most recent data for evaluation.  

      ![Data Splitting](https://github.com/user-attachments/assets/4890909f-c3c7-41bf-b97c-669e3033f895)
      
## 8. Model Development
My approach involves testing and comparing several types of models to determine the best fit for cryptocurrency price prediction:

- Machine Learning Models: Random Forest & XGBoost
  - Capture complex, non-linear relationships
- Time Series Models: ARIMA, SARIMA & Prophet
  - Model trends, seasonality, and cyclical patterns in the data.
- Deep Learning Models: Long Short Term Memory (LSTM)
  - Leverage recurrent neural networks to capture long-term dependencies in    sequential data.

## Machine Learning Models

### Random Forest
- Random Forest is an ensemble learning algorithm that builds multiple decision trees and merges their predictions to improve accuracy and reduce overfitting. It is used for both **classification** and **regression** tasks.
- Random Forest follows a **bagging (Bootstrap Aggregation) approach** to create multiple decision trees and combines their outputs for better predictions.

#### How Random Forest Works

1. **Bootstrap Sampling:**  
   - The dataset is randomly sampled **with replacement** to create multiple subsets (bootstrap samples).  
   - Each subset is used to train a separate decision tree.  

2. **Feature Selection at Each Split:**  
   - Instead of considering all features, Random Forest selects a **random subset of features** for each split in a tree.  
   - This introduces variability, making trees less correlated and reducing overfitting.  

3. **Grow Multiple Decision Trees:**  
   - Each tree is trained independently on a different bootstrap sample.  
   - The trees are **fully grown** without pruning.  

4. **Aggregation of Predictions:**  
   - **For classification:** The final prediction is made using **majority voting** (the most common class among trees).  
   - **For regression:** The final prediction is the **average of all tree outputs**.  

---

<img src="https://github.com/user-attachments/assets/7578abd1-e7e2-4fbb-92d3-7c6bb9165d83" style="width: 80%; height: auto;">

 Hyperparameters
- n_estimators = 200 trees for better accuracy and reduced variance.
- max_depth = 10: Limits tree depth to prevent overfitting.
- max_features = 'sqrt': Randomly selects features to add diversity and reduce overfitting.
- random_state = 42: Ensures reproducibility and consistency.

### XGBoost (eXtreme Gradient Boosting)
- XGBoost builds an ensemble of decision trees sequentially, where each new tree corrects the errors of the previous ones using gradient descent.
- Gradient Descent is an optimization algorithm used in machine learning to minimize the loss function by iteratively adjusting model parameters.

#### How XGBoost Works:
1. **Initialize Predictions:**  
   - The first prediction is a simple base value, such as the mean (for regression) or a uniform probability (for classification).

2. **Compute Loss Function:**  
   - A loss function (e.g., Mean Squared Error for regression, Log Loss for classification) measures how far predictions are from actual values.

3. **Compute Gradients (First-Order Derivatives) and Hessians (Second-Order Derivatives):**  
   - XGBoost calculates the gradient (first derivative of the loss) to determine the direction of improvement.
   - It also computes the Hessian (second derivative) to estimate the step size for updates.

4. **Build Decision Trees:**  
   - New trees are added iteratively to minimize the loss.
   - Trees are constructed using a greedy algorithm, selecting the best feature splits based on gain.
   - Instead of directly predicting target values, trees predict adjustments (residuals) to previous predictions.

5. **Weight Updates:**  
   - Leaf values of trees are updated using the computed gradients and Hessians.
   - Learning rate (shrinkage factor) controls the contribution of each tree to prevent overfitting.

6. **Regularization & Pruning:**  
   - XGBoost includes L1 (Lasso) and L2 (Ridge) regularization to reduce overfitting.
   - It prunes trees by removing splits that do not contribute significantly to reducing loss.

7. **Final Prediction:**  
   - The predictions from all trees are summed up, adjusting for the learning rate.

---
<img src="https://github.com/user-attachments/assets/7f83aa96-ceb1-4e19-ae42-5f259b9ac26f" style="width: 80%; height: auto;">

Hyperparameters
- n_estimators = 200 trees for improved accuracy.
- learning_rate = 0.05 to prevent overfitting.
- max_depth = 7 to control tree complexity.
- L2 (0.5) ridge regularization to prevent any single feature from dominating the model, since the dataset has highly correlated features.
- Random state of 42 for consistent results.

#### Machine learning Model Evaluation
<!-- First Image Pair -->
<div style="display: flex; justify-content: center; align-items: center; gap: 10px;">
  <img src="https://github.com/user-attachments/assets/0af46f87-e27c-4baa-b44a-d79e1cee3841" width="45%">
  <img src="https://github.com/user-attachments/assets/e8f5ea9a-fd5f-44e5-9754-979c4f3c9f59" width="45%">
</div>

<!-- Second Image Pair -->
<div style="display: flex; justify-content: center; align-items: center; gap: 10px; margin-top: 10px;">
  <img src="https://github.com/user-attachments/assets/40af424a-7389-4a6e-b47f-c9ebc3d735ee" width="30%">
  <img src="https://github.com/user-attachments/assets/4e5a8b74-43bb-4533-9606-ddbb26f9f4d0" width="30%">
</div>



#### Model Predictions
<div style="display: flex; justify-content: center;">
  <img src="https://github.com/user-attachments/assets/5b9ef78b-b361-4043-94e6-e9d1441ddb13" width="45%" style="margin-right: 10px;">
  <img src="https://github.com/user-attachments/assets/5055fd2e-f355-4e03-94ad-f0ab7982f206" width="45%">
</div>


## Time Series Models
### Autoregressive Integrated Moving Average (ARIMA)

The **ARIMA** model is a popular statistical method used for time series forecasting. It captures patterns in time series data by combining three components.

**1. AutoRegressive (AR) Component (p)**
- Uses past values of the series to predict the next value.
- If `p = 2`, the current value depends on the last two values:

**2. Integrated (I) Component (d)**
- Ensures the time series is stationary (constant mean and variance over time).
- Differencing is applied \( d \) times until the series becomes stationary.

**3. Moving Average (MA) Component (q)**
- Uses past forecast errors to improve predictions.
- If `q = 1`, the model includes the last error term:

#### How ARIMA Works

1. **Make the Series Stationary**  
   - Check for trends or seasonality.
   - Apply differencing if necessary.

2. **Select ARIMA Parameters (p, d, q)**  
   - Use **Autocorrelation Function (ACF)** and **Partial Autocorrelation Function (PACF)** plots.

3. **Fit the Model**  
   - Train ARIMA on historical data.

4. **Make Forecasts**  
   - Once trained, use the model to predict future values.
  
### SARIMA (Seasonal AutoRegressive Integrated Moving Average)

The **SARIMA (Seasonal ARIMA)** model is an extension of the **ARIMA** model that accounts for **seasonality** in time series data. It is represented as:

\[
SARIMA(p, d, q) \times (P, D, Q, s)
\]

where:  
- **(p, d, q)** are the ARIMA components:
  - **p** = AutoRegressive (AR) order (lags of past values)
  - **d** = Differencing order (for stationarity)
  - **q** = Moving Average (MA) order (lags of past errors)
- **(P, D, Q, s)** are the seasonal components:
  - **P** = Seasonal AutoRegressive order
  - **D** = Seasonal differencing order
  - **Q** = Seasonal Moving Average order
  - **s** = Length of the seasonal cycle (e.g., `12` for monthly data)

---

#### **How SARIMA Works**

**1. Differencing for Stationarity**
SARIMA applies **two types of differencing** to make the time series stationary:
- **Regular differencing (d):** Removes trends in the data.
- **Seasonal differencing (D):** Removes repeating seasonal patterns.

For example, if working with monthly data, a **seasonal difference** would be:

\[
Y_t' = Y_t - Y_{t-s}
\]

where \( s \) is the season length (e.g., 12 for monthly data).

**2. Handling Seasonality with SARIMA**
The seasonal components **(P, D, Q, s)** work similarly to the non-seasonal components, but they operate at the **seasonal level** instead of individual time steps.

For example, in a **SARIMA(1,1,1)(1,1,1,12)** model for **monthly data**:
- **(1,1,1):** ARIMA terms handling short-term dependencies.
- **(1,1,1,12):** Seasonal ARIMA terms handling annual seasonal patterns.

**3. Model Selection**
- Use **Autocorrelation Function (ACF)** and **Partial Autocorrelation Function (PACF)** plots to choose \( p, q, P, Q \).
- Use **Akaike Information Criterion (AIC)** to find the best parameter combination.

**4. Forecasting with SARIMA**
Once fitted, SARIMA can generate forecasts that **incorporate seasonal patterns** along with trends.

---

![image](https://github.com/user-attachments/assets/25f53d72-2a9b-4ab1-adae-0da933d0c08d)

**Checking for Stationarity**

- **Visual Inspection (Rolling Mean & Standard Deviation):**  
  ![Rolling Statistics](https://github.com/user-attachments/assets/023abafb-c221-466c-b3b5-ded6b156e080)
  
- **ADF Test Results:**  
  - **Original Series:**  
    - ADF Statistic: 0.2040  
    - p-value: 0.9725  
    → *Non-stationary*
    
**Transformations for Stationarity**

- **Log Transformation:**
  - Applied log transformation to stabilize the variance and reduce the effect of large fluctuations in the data.
  - This helped smooth out exponential growth trends.

- **Differencing:**  
  - Applied first-order differencing to remove trends in the data.
  - Dickey-Fuller (ADF) Test to confirm that the transformed data met the stationarity assumption.

<div style="display: flex; justify-content: center; align-items: center; gap: 10px;">
  <img src="https://github.com/user-attachments/assets/95046dac-709b-421d-bd69-ff08dac9b9d7" width="45%">
  <img src="https://github.com/user-attachments/assets/d2747dd4-f524-4a60-aaf8-6416b2a8ca0d" width="45%">
</div>

- **Differenced Series:**  
  - ADF Statistic: -62.8500  
  - p-value: 0.0000  
  → *Stationary*

**Autocorrelations (ACF) and Partial Autocorrelations (PACF) Plot**
![image](https://github.com/user-attachments/assets/37769808-1563-4172-a7e0-dd2d91981372)

![image](https://github.com/user-attachments/assets/a7deada9-4cda-48a6-b6f1-caa1d6788761)

- ACF shows a sharp drop after lag 1 suggesting (Moving Average order = 1).
  
![image](https://github.com/user-attachments/assets/8f1c534f-5d3e-4480-82f6-68cc8e45eab9)

- PACF cuts off after lag 1 suggesting  (Autoregressive order = 1).
-  The ACF and PACF plot drops off quickly (with no strong pattern or slow decay), meaning the data is stationary after differencing.
-  This confirms that the differencing step (d = 1) was effective in removing trends.

**Seasonality Analysis:**  

<div style="display: flex; justify-content: center; align-items: center; gap: 10px;">
  <img src="https://github.com/user-attachments/assets/501b8ce2-6f36-487a-9114-0955f8219bef" width="45%">
  <img src="https://github.com/user-attachments/assets/dca441b1-4044-485f-b68e-8ee47b71feb0" width="45%">
</div>

- Before differencing, there is a clear upward trend, but after differencing, the trend became more stable with gradual fluctuations.
- There is a strong seasonal pattern and afterwards the repeating pattern is still visible, indicating that seasonality remains present.
- Additionally, the residuals appear more stationary after differencing, suggesting that the data is closer to meeting the assumptions of stationarity.

#### ARIMA Model Development

<img src="https://github.com/user-attachments/assets/c937d402-72b4-4883-ac75-8e6b953804c4" width="60%">

- AR(1) coefficient 0.9996: Strong positive influence of past values on future values. 
- MA(1) coefficient -0.0180: Minimal negative effect of past errors on the current value. 
- sigma² (variance of residuals): Very small (0.0015), indicating a well-fitted model.

#### SARIMA Model Development

<img src="https://github.com/user-attachments/assets/1999f176-c3c0-49f5-ac79-a154b3ffb5a6" width="60%">

- AR(1) coefficient 1.0001: Strong positive influence of past values on future values.
- MA(1) coefficient -0.0185: Small negative effect of past errors on the current value.
- Seasonal AR(12) coefficient -0.1530: Weak inverse seasonal dependency.
- Seasonal MA(12) coefficient 0.1442: Weak positive seasonal effect.

### Model Predictions and Evaluation
<!-- First Row (Side by Side) -->
<div style="display: flex; justify-content: center; gap: 10px; align-items: center;">
  <img src="https://github.com/user-attachments/assets/64f024d1-bcef-46bb-af90-cff3a5e39c22" style="width: 45%; height: auto;">
  <img src="https://github.com/user-attachments/assets/e82d0dbf-d234-4b7a-8d0b-a0c0d8e089fe" style="width: 45%; height: auto;">
</div>

<!-- Second Row (Side by Side) -->
<div style="display: flex; justify-content: center; gap: 10px; align-items: center; margin-top: 10px;">
  <img src="https://github.com/user-attachments/assets/1202f095-79e5-490d-b98e-bdfe405459b3" style="width: 20%; height: auto;">
  <img src="https://github.com/user-attachments/assets/b1f86c0f-8315-4ca5-8333-ae93b9b895d2" style="width: 20%; height: auto;">
</div>


### FB Prophet (Facebook Prophet)
- Prophet is an **open-source time series forecasting model** developed by Facebook (Meta).
- It is designed to handle missing data, seasonality, and trend shifts, making it highly effective for business and financial forecasting.

#### How Prophet Works

Prophet models time series data as a combination of the following components:

\[
y(t) = g(t) + s(t) + h(t) + \epsilon_t
\]

Where:
- **\( g(t) \)**: Trend component (captures long-term growth or decline).
- **\( s(t) \)**: Seasonal component (captures weekly, monthly, or yearly seasonality).
- **\( h(t) \)**: Holiday effects (captures special events that impact the time series).
- **\( \epsilon_t \)**: Error term (captures any noise in the data).

#### Step-by-Step Process:

1. **Trend Estimation:**  
   - Prophet fits a **piecewise linear or logistic growth model** to capture the overall trend.  
   - It detects **change points** where the trend shifts.  

2. **Seasonality Modeling:**  
   - Uses **Fourier series** to model periodic seasonal variations.  
   - Automatically detects daily, weekly, and yearly patterns.  

3. **Holiday Effects:**  
   - Users can define a list of holidays/events that impact the time series.  
   - The model assigns additional effects for these events.  

4. **Uncertainty Intervals:**  
   - Prophet generates **uncertainty intervals** around forecasts to account for variability.  

5. **Prediction Generation:**  
   - Once trained, the model generates future predictions, adjusting for trend, seasonality, and holiday effects.  

---

#### Prohet model evaluation and predictions
<div style="display: flex; justify-content: center; gap: 10px;">
  <img src="https://github.com/user-attachments/assets/3e01e2ad-606e-4522-b312-78fafbbbc799" style="width: 48%; height: auto;">
  <img src="https://github.com/user-attachments/assets/3ad97d31-b6bd-4cf9-a9f2-9b00dde24d9d" style="width: 48%; height: auto;">
</div>

<img src="https://github.com/user-attachments/assets/461c361b-0023-4fa2-94c0-760fe20784fd" width="20%">

### Recurrent Neural Networks (RNN) and LSTM:

- A **Recurrent Neural Network (RNN)** is a type of **neural network** designed for **sequential data**.
- Unlike traditional neural networks, which assume inputs are independent of each other, RNNs **retain memory of previous inputs** to make better predictions for current inputs.

#### How RNNs Work:
At each time step \( t \), the RNN takes an input \( x_t \) and the previous hidden state \( h_{t-1} \), then computes the new hidden state \( h_t \):

\[
[h_t = f(W_x x_t + W_h h_{t-1} + b)\]
\]

where:
- (h_t\) represents the hidden state at time step \(t\)
- (x_t\) denotes the input at time step \(t\)
- (W_x\) corresponds to the weight matrix for input
- (W_h\) stands for the weight matrix for the hidden state
- (b\) symbolizes the bias vector
- (f\) denotes a non-linear activation function (typically hyperbolic tangent, tanh, or rectified linear unit, ReLU)

<img src="https://github.com/user-attachments/assets/51c7e8d5-010f-49ef-a9dc-5b91b508ee40" style="width: 80%; height: auto;">

#### Problem with RNNs: Vanishing Gradient
When training deep RNNs, the gradients during backpropagation **become extremely small (vanish)**, making it hard for the network to remember long-term dependencies. This is known as the **vanishing gradient problem**.

### Long Short-Term Memory (LSTM)

- **LSTM (Long Short-Term Memory)** is a type of **Recurrent Neural Network (RNN)** designed to handle long-term dependencies in **sequential data**.
- It overcomes the **vanishing gradient problem** faced by traditional RNNs by using a memory cell and three gates (**Forget, Input, Output**).

#### LSTM Architecture

Each LSTM cell contains:
1. **Forget Gate (\( f_t \))** – Decides what information should be discarded.
2. **Input Gate (\( i_t \))** – Decides what new information to store.
3. **Cell State (\( C_t \))** – Stores important long-term information.
4. **Output Gate (\( o_t \))** – Decides what information should be sent to the next time step.

---

#### How LSTM Works

At each time step \( t \), LSTM takes **input \( x_t \)** and the **previous hidden state \( h_{t-1} \)** and processes them through the following steps:

**Step 1: Forget Gate (\( f_t \))**
- Determines how much of the previous memory \( C_{t-1} \) should be kept.
- Uses a **sigmoid activation function** (\( \sigma \)) to output a value between **0 (forget everything)** and **1 (keep everything)**.

\[
f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)
\]

Where:
- \( W_f \) = Weight matrix for forget gate.
- \( b_f \) = Bias term.
- \( h_{t-1} \) = Previous hidden state.
- \( x_t \) = Current input.
- \( \sigma \) = Sigmoid activation function.

---

**Step 2: Input Gate (\( i_t \))**
- Decides what new information should be added to the cell state.
- Uses **two functions**:
  1. A **sigmoid function** to decide which values to update.
  2. A **tanh function** to create a vector of new candidate values.

\[
i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)
\]

\[
\tilde{C_t} = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)
\]

Where:
- \( W_i \) and \( b_i \) = Weights and bias for the input gate.
- \( W_C \) and \( b_C \) = Weights and bias for new candidate values.
- \( \tilde{C_t} \) = New candidate values.

---

**Step 3: Update Cell State (\( C_t \))**
- The cell state is updated using:
  - The **forget gate** to remove old information.
  - The **input gate** to add new information.

\[
C_t = f_t * C_{t-1} + i_t * \tilde{C_t}
\]

Where:
- \( C_{t-1} \) = Previous cell state.
- \( C_t \) = Updated cell state.

---

**Step 4: Output Gate (\( o_t \))**
- Determines what the next **hidden state \( h_t \)** should be.
- Uses a **sigmoid function** to filter the important parts of the cell state.
- Passes the filtered cell state through a **tanh function**.

\[
o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)
\]

\[
h_t = o_t * \tanh(C_t)
\]

Where:
- \( W_o \) and \( b_o \) = Weights and bias for the output gate.
- \( h_t \) = Updated hidden state (passed to the next time step).

---

<img src="https://github.com/user-attachments/assets/4f0a13c8-ba72-4406-981c-624e1298d3b9" style="width: 80%; height: auto;">

#### LSTM Model Development

<img src="https://github.com/user-attachments/assets/53f7d205-bf38-4adb-bc2e-0ca4009c7baa" style="width: 60%; height: auto;">

- **LSTM Layers:**  
  - **Layer 1:** 50 units, returns sequences, uses tanh activation with L2 regularization.
  - **Layer 2:** 40 units, final output state, tanh activation with L2 regularization.
  
- **Dropout Layers:**  
  0.3 dropout after each LSTM layer to mitigate overfitting.
  
- **Dense Layers:**  
  - 30-unit dense layer with ReLU activation.
  - Final dense layer with 1 unit for regression.

- **Training Details:**  
  - **Optimizer:** Adam with a learning rate of 0.0005  
  - **Loss Function:** Mean Squared Error (MSE)  
  - **Epochs:** Up to 100 with early stopping (patience = 10, restore best weights)
    
- Trained separate models for Bitcoin, Ethereum, and Solana using a batch size of 64 over up to 100 epochs, storing training histories for analysis.

#### LSTM Model Performance

![image](https://github.com/user-attachments/assets/e6c274ea-b5dd-44d2-8056-807f89b3d4cf)

- The loss curves show a steady and consistent decrease, indicating that the models are learning effectively.
- The training and validation loss curves are closely aligned, meaning the models generalize well to unseen data.
- The models reach a near-zero plateau, suggesting they have converged and additional training may not bring significant improvements.

#### LSM Model Predictions

<div style="display: flex; flex-wrap: wrap; justify-content: center; gap: 15px;">
  <img src="https://github.com/user-attachments/assets/08ad6e78-d58a-4db9-ad0d-e553f4bf3558" style="width: 49%; height: 300px; object-fit: cover;">
  <img src="https://github.com/user-attachments/assets/a0da0b5b-a504-40c4-acf5-109a83fc0b35" style="width: 49%; height: 300px; object-fit: cover;">
  <img src="https://github.com/user-attachments/assets/c8ff1783-cdad-41f5-ba78-1cbf219cff51" style="width: 49%; height: 300px; object-fit: cover;">
</div>
 
**LSTM Crypto model evaluation**

| **Model**          | **MAE**    | **RMSE**   |
|--------------------|------------|------------|
| **Bitcoin**        | 2147.02    | 2832.39    |
| **Ethereum**       |  125.98    | 174.23     |
| **Solana**         |  10.47     | 13.71      |


## 10. Model Evaluation

### **Metrics**

- **Mean Absolute Error (MAE):**  
  - Represents the average absolute difference between predicted and actual values.
  - Less sensitive to outliers.

- **Root Mean Squared Error (RMSE):**  
  - Gives higher weight to larger errors.
  - In the same units as the target variable, allowing direct comparison.

### **Performance Summary**

| **Model**          | **MAE**    | **RMSE**   |
|--------------------|------------|------------|
| **LSTM**           | 2147.02    | 2832.39    |
| **Random Forest**  | 4761.72    | 11494.35   |
| **XGBoost**        | 5357.49    | 12270.71   |
| **Prophet**        | 11342.16   | 15621.58   |
| **ARIMA**          | 26449.34   | 34524.33   |
| **SARIMA**         | 15283.65   | 20759.14   |

**Conclusion:**  
Based on these metrics, **LSTM** emerges as the best-performing model for forecasting Bitcoin prices with the lowest MAE and RMSE values.

## 11. Challenges
- Extreme Market Volatility:
  - Rapid, large price swings make it challenging to establish stable trends and reliable predictions.
- Regulatory Uncertainty:
  - Shifts in government policies can cause sudden market changes, adding significant unpredictability.
- Influence of Market Sentiment:
  - Public opinion, driven by social media and news, can unexpectedly affect prices, complicating prediction models.

## 12. Model Deployment: 
**Streamlit app**

![image](https://github.com/user-attachments/assets/a2693de9-a311-4055-b01f-042ae69ae9fb)

- Streamlit link: https://share.streamlit.io/

## 13. Further Development
- Integrate Market Sentiment Analysis:
  - Incorporate real-time sentiment data from social media and news, using NLP techniques to gauge public mood as an early indicator of market shifts.
- Real-Time Data Processing:
  - Setting up a robust pipeline to ensure that the model updates every hour, capturing the latest market data without delays.
- Visualization & Monitoring:
  - Lastly, I'll develop a dashboard to visualize real-time predictions, sentiment trends, and key performance metrics, allowing me to monitor and refine the model effectively.


# Cryptocurrency Price Prediction - Model Performance Analysis  

## Summary of Findings  
After updating the dataset to use a **shorter time period (2020-2025)** and testing on the last **two months of 2025**, I evaluated multiple models to determine their effectiveness in predicting Bitcoin prices. Additionally, I performed **feature engineering** to enhance model performance.  

## Model Performance Comparison  

| Model    | Mean Absolute Error (MAE) | Root Mean Squared Error (RMSE) | R² Score |
|----------|--------------------------|-------------------------------|----------|
| **Random Forest** | 2474.29 | 3564.90 | 0.7840 |
| **XGBoost** | **1923.29** | **2606.91** | **0.8845** |
| **ARIMA** | 11926.59 | 14639.69 | -2.6433 |
| **SARIMA** | 13548.57 | 16473.83 | -3.6134 |
| **PROPHET** | 13862.20 | 16943.50 | -3.8824 |
| **LSTM** | 2545.40 | 3205.66 | 0.8115 |

### **Key Observations:**  
- **XGBoost emerged as the best-performing model**, achieving the lowest MAE (1923.29) and RMSE (2606.91), with the highest R² score (0.8845).  
- **LSTM was the second-best model**, with a strong R² score (0.8115) but slightly higher errors compared to XGBoost.  
- **Random Forest performed reasonably well** (R² = 0.7840) but was outperformed by XGBoost and LSTM.  
- **Traditional time series models (ARIMA, SARIMA, Prophet) failed to capture trends effectively**, producing negative R² scores.  

## Feature Engineering  
To improve model accuracy, I incorporated various technical indicators and trading volume data, including:  
- **Moving Averages (MA)**  
- **Relative Strength Index (RSI)**  
- **Moving Average Convergence Divergence (MACD)**  
- **Bollinger Bands**  
- **Trading Volume**  

These features provided additional insights into price trends and market momentum, significantly impacting model performance.  

## Insights from the Shorter Time Span  
- In the previous experiment (2014-2023), **LSTM was the best-performing model**. However, in this updated analysis with a shorter time span, **XGBoost outperformed LSTM**.  
- This suggests that for **shorter-term price predictions, tree-based models like XGBoost may be more effective** than deep learning models.  
- **Time series models struggled significantly**, indicating that Bitcoin’s price movements may not follow simple time-based patterns.  

## Addition of More Cryptocurrencies  
To enhance model performance and explore multi-coin dependencies, I expanded the dataset to include the following cryptocurrencies:  
- **Bitcoin (BTC)**  
- **Ethereum (ETH)**  
- **Solana (SOL)**  
- **Tether (USDT)**  
- **XRP (XRP)**  
- **Binance Coin (BNB)**  

This allows for potential improvements in predictive accuracy by analyzing interdependencies between different cryptocurrencies.  

## Next Steps  
To further improve the predictions, I plan to:  
✅ **Fine-tune the LSTM model** by adjusting hyperparameters (e.g., number of layers, learning rate, dropout, and epochs) to see if performance can be improved.  
✅ **Further optimize XGBoost** to determine if its performance can be enhanced with additional tuning.  

---
This analysis provides valuable insights into the performance of different models across different time frames, and future improvements will focus on optimizing feature selection and model tuning.
