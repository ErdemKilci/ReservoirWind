# 🔋 Reservoir Wind - AI-Based Power Consumption Forecasting

**Authors:** Said Aydin, Sami Cenkci, Erdem Kilci  
**Supervisor:** Alexander Stasik (SINTEF)  
**Institution:** OsloMet – Oslo Metropolitan University  
**Date:** Spring 2025

---

## 📌 Project Overview

**Reservoir Wind** is a bachelor thesis project aimed at building an AI-based forecasting system that predicts **power consumption using historical load data**. This project combines classical time-series modeling with modern deep learning and reservoir computing techniques to produce a reliable, interpretable, and scalable solution.

The project is developed in collaboration with **SINTEF**, one of Europe’s leading independent research organizations, and is supervised by **Alexander Stasik**, a specialist in Reservoir Computing.

---

## 🎯 Problem Statement

Forecasting power consumption is a critical task for ensuring **grid stability** and reducing **energy waste**. However, existing methods like ARIMA/SARIMA often fall short due to:

- Poor handling of non-linear, chaotic systems
- Stationarity assumptions that don't hold in real-world data
- Limited real-time adaptability

Our solution addresses these challenges using a multi-model machine learning approach.

---

## 💡 Objectives

- Engineer time-aware and error-sensitive features
- Build and evaluate several forecasting models:
  - **Echo State Networks (ESNs)** via Reservoir Computing
  - **LSTM** and **CNN+LSTM** deep learning architectures
  - **Prophet** with additional regressors and seasonality tuning
- Automate hyperparameter tuning using randomized search with caching and cross-validation
- Evaluate model performance with comprehensive metrics and visual diagnostics

---

## 🧰 Tools & Technologies

- **Python 3.10+**
- Data Processing: `pandas`, `numpy`, `matplotlib`, `seaborn`
- Forecasting Models:
  - `ReservoirPy` (ESN)
  - `PyTorch` (LSTM, CNN+LSTM)
  - `prophet` (time series with regressors)
- Evaluation: `scikit-learn`, `scipy`
- Caching & Optimization: `joblib`, `ParameterSampler`, `TimeSeriesSplit`
- Visualizations: Line plots, error histograms, autocorrelation, residuals, correlation heatmaps

---

## 🧼 Data Preprocessing & Feature Engineering

We handle missing values with a hybrid method:

- **Short gaps**: Cubic interpolation
- **Longer gaps**: Hourly imputation from previous weeks (same weekday & hour)

### Extracted Features:

- **Time Features**: Hour, Day, Month, Weekday, Weekend
- **Lagged Values**: 1H, 24H, and 1-Year lag for both Forecast and Actual Load
- **Rolling Means**: 6H, 24H, 1W, 1M windows
- **Error Features**: Load Error, Absolute Error, MAPE

All features are normalized with `MinMaxScaler`.

---

## 🤖 Forecasting Models

### 🔹 Echo State Network (ESN)

- Implemented using ReservoirPy
- Random search over number of units, spectral radius, leak rate, input scaling
- TimeSeriesSplit cross-validation (3-fold)
- Caching for efficient experimentation

### 🔹 LSTM & CNN+LSTM

- Implemented in PyTorch with caching and repeated seed runs
- Hyperparameters: hidden size, number of layers, learning rate, batch size
- CNN+LSTM includes a 1D convolution before LSTM layer
- Results averaged across multiple seeds

### 🔹 Prophet

- Regressors: hour, weekday, month, lag_1H, lag_24H
- Random search over changepoint prior, seasonality prior, Fourier orders
- Forecast decomposition plots and uncertainty intervals

---

## 📊 Evaluation & Visualization

Each model is evaluated on a held-out test set using:

- **MAE** – Mean Absolute Error
- **RMSE** – Root Mean Squared Error
- **MAPE** – Mean Absolute Percentage Error
- **R² Score**

Additional analysis:

- **Paired t-tests** (ESN vs others)
- **Error distributions** (histograms & boxplots)
- **Residual autocorrelation**
- **Actual vs Predicted scatter plots**
- **Model ranking table sorted by MAPE**

All plots are stored in `/plots/` and `/ReportFigures/`.

---

## 🏆 Final Results

| Model      | MAE     | RMSE    | MAPE (%) | R² Score |
|------------|---------|---------|----------|----------|
| **ESN**    | 140.86  | 185.02  | **0.94** | **0.9967** |
| LSTM       | 221.37  | 295.69  | 1.47     | 0.9915   |
| CNN+LSTM   | 207.08  | 269.34  | 1.36     | 0.9930   |
| Prophet    | 171.68  | 229.08  | 1.16     | 0.9949   |

---

## 🗂️ Project Structure

```text
├── Data/                        # Raw data 
├── joblib_cache/               # Caching directory for model training
├── plots/                      # All result visualizations and evaluation plots
├── reservoir_wind.ipynb        # Jupyter/Python files for each pipeline step
├── ReportFigures/              # Figures used in the final project report
├── README.md                   # This file
├── 2015&2024/                  # Raw data from 2014 & 2025
└── combined_data.csv           # Combined and cleaned data from 2014 to 2024
