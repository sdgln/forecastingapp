import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(page_title="📈 Sales Forecasting App", layout="wide")
st.title("📊 Sales Forecasting Interface")

uploaded_file = st.file_uploader("Upload your sales CSV file with 'Order Date' and 'Sales' columns")

def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return mae, rmse, mape

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df['Order Date'] = pd.to_datetime(df['Order Date'], dayfirst=True)
    df = df.sort_values('Order Date')
    df.set_index('Order Date', inplace=True)

    # Monthly aggregation
    monthly_sales = df.resample('M').sum()['Sales']
    st.subheader("Raw Monthly Sales")
    st.line_chart(monthly_sales)

    # Outlier handling
    Q1, Q3 = monthly_sales.quantile([0.25, 0.75])
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    cleaned = monthly_sales.copy()
    cleaned[(monthly_sales < lower) | (monthly_sales > upper)] = monthly_sales.mean()

    st.subheader("Outlier-Cleaned Sales")
    st.line_chart(cleaned)

    # Moving Averages
    ma3 = cleaned.rolling(3).mean()
    ma6 = cleaned.rolling(6).mean()
    st.subheader("Moving Averages")
    fig, ax = plt.subplots()
    ax.plot(cleaned, label="Actual")
    ax.plot(ma3, label="MA(3)")
    ax.plot(ma6, label="MA(6)")
    ax.legend()
    ax.grid()
    st.pyplot(fig)

    # Evaluate MA(3)
    y_true = cleaned[3:]
    y_pred_ma3 = ma3[3:]
    mae, rmse, mape = calculate_metrics(y_true, y_pred_ma3)
    st.write("**MA(3) Performance**")
    st.write(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}%")

    # Evaluate MA(6)
    y_true = cleaned[6:]
    y_pred_ma6 = ma6[6:]
    mae, rmse, mape = calculate_metrics(y_true, y_pred_ma6)
    st.write("**MA(6) Performance**")
    st.write(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}%")

    # SES
    ses_model = SimpleExpSmoothing(cleaned).fit(smoothing_level=0.2, optimized=False)
    ses_forecast = ses_model.fittedvalues
    st.subheader("Simple Exponential Smoothing")
    fig, ax = plt.subplots()
    ax.plot(cleaned, label="Actual")
    ax.plot(ses_forecast, label="SES Forecast")
    ax.legend()
    ax.grid()
    st.pyplot(fig)

    # SES Metrics
    y_true = cleaned[1:]
    y_pred_ses = ses_forecast[1:]
    mae, rmse, mape = calculate_metrics(y_true, y_pred_ses)
    st.write("**SES Performance**")
    st.write(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}%")

    # Holt-Winters
    hw_model = ExponentialSmoothing(cleaned, trend="add", seasonal="add", seasonal_periods=12).fit()
    hw_forecast = hw_model.fittedvalues
    st.subheader("Holt-Winters Forecast")
    fig, ax = plt.subplots()
    ax.plot(cleaned, label="Actual")
    ax.plot(hw_forecast, label="Holt-Winters")
    ax.legend()
    ax.grid()
    st.pyplot(fig)

    y_true = cleaned[12:]
    y_pred_hw = hw_forecast[12:]
    mae, rmse, mape = calculate_metrics(y_true, y_pred_hw)
    st.write("**Holt-Winters Performance**")
    st.write(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}%")

    # Machine Learning: Random Forest
    df_ml = cleaned.to_frame()
    df_ml['lag1'] = df_ml['Sales'].shift(1)
    df_ml['lag2'] = df_ml['Sales'].shift(2)
    df_ml.dropna(inplace=True)
    X = df_ml[['lag1', 'lag2']]
    y = df_ml['Sales']
    X_train, X_test = X.iloc[:int(0.8*len(X))], X.iloc[int(0.8*len(X)):]
    y_train, y_test = y.iloc[:int(0.8*len(X))], y.iloc[int(0.8*len(X)):] 

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', RandomForestRegressor())
    ])
    param_grid = {
        'model__n_estimators': [100],
        'model__max_depth': [5, 10, None]
    }
    search = GridSearchCV(pipeline, param_grid, cv=3, scoring='neg_mean_squared_error')
    search.fit(X_train, y_train)
    best_model = search.best_estimator_

    y_pred = best_model.predict(X_test)

    st.subheader("Random Forest Forecast vs Actual")
    fig, ax = plt.subplots()
    ax.plot(y_test.index, y_test, label="Actual", marker='o')
    ax.plot(y_test.index, y_pred, label="Random Forest", linestyle='--')
    ax.legend()
    ax.grid()
    st.pyplot(fig)

    st.subheader("📉 Random Forest Model Performance")
    mae, rmse, mape = calculate_metrics(y_test, y_pred)
    st.write(f"MAE: {mae:.2f}")
    st.write(f"RMSE: {rmse:.2f}")
    st.write(f"MAPE: {mape:.2f}%")
else:
    st.info("📂 Please upload a CSV file with 'Order Date' and 'Sales' columns.")

