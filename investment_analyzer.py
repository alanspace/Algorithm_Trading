import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

import tensorflow as tf

print(f"TensorFlow version: {tf.__version__}")

# --- THIS IS THE CORRECTED GPU CHECK ---
# The modern way to check for a GPU in TensorFlow 2.13+
if tf.config.list_physical_devices('GPU'):
    print("GPU device is available.")
    # You can also see the device name and details
    print("GPU device details:", tf.config.list_physical_devices('GPU'))
else:
    print("GPU device is not available. TensorFlow will use the CPU.")
# ----------------------------------------


# --- Configuration ---
# You can change these tickers to any you are interested in
TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
START_DATE = (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d') # 5 years of data
END_DATE = datetime.now().strftime('%Y-%m-%d')
LOOKBACK_DAYS = 60  # Number of past days' data to use for prediction

def fetch_data(tickers, start, end):
    """Fetches historical stock data from Yahoo Finance."""
    print("Fetching stock data...")
    stock_data = yf.download(tickers, start=start, end=end)['Close']
    print("Data fetched successfully.")
    return stock_data.dropna()

def create_lstm_model(input_shape):
    """Builds and compiles an LSTM model."""
    model = Sequential([
        LSTM(100, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def predict_trends(data):
    """
    Trains an LSTM model for each stock and predicts the next day's price movement.
    """
    print("\n--- Predicting Future Trends with LSTM ---")
    predictions = {}

    for ticker in data.columns:
        print(f"\nProcessing {ticker}...")
        
        # 1. Prepare Data
        scaler = MinMaxScaler(feature_range=(0,1))
        scaled_data = scaler.fit_transform(data[[ticker]])

        X, y = [], []
        for i in range(LOOKBACK_DAYS, len(scaled_data)):
            X.append(scaled_data[i-LOOKBACK_DAYS:i, 0])
            y.append(scaled_data[i, 0])
        
        X_train, y_train = np.array(X), np.array(y)
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

        # 2. Build and Train Model
        model = create_lstm_model((X_train.shape[1], 1))
        model.fit(X_train, y_train, batch_size=32, epochs=25, verbose=0)

        # 3. Predict Next Day
        last_sequence = scaled_data[-LOOKBACK_DAYS:]
        last_sequence = np.reshape(last_sequence, (1, LOOKBACK_DAYS, 1))
        predicted_price_scaled = model.predict(last_sequence)
        predicted_price = scaler.inverse_transform(predicted_price_scaled)[0][0]

        # 4. Determine Trend
        last_actual_price = data[ticker].iloc[-1]
        trend = "Up" if predicted_price > last_actual_price else "Down"
        predictions[ticker] = {
            'Predicted Price': predicted_price,
            'Last Actual Price': last_actual_price,
            'Trend': trend
        }
        print(f"Prediction for {ticker}: Trend is '{trend}' (Predicted: ${predicted_price:.2f}, Last Actual: ${last_actual_price:.2f})")
    
    return predictions

def perform_pca_risk_analysis(data):
    """
    Performs Principal Component Analysis (PCA) to identify main risk factors.
    """
    print("\n--- Performing Risk Analysis with PCA ---")
    
    # Calculate daily returns
    daily_returns = data.pct_change().dropna()
    
    # Normalize returns
    scaler = MinMaxScaler()
    scaled_returns = scaler.fit_transform(daily_returns)
    
    # Perform PCA
    pca = PCA()
    pca.fit(scaled_returns)
    
    # Analyze the principal components
    explained_variance_ratio = pca.explained_variance_ratio_
    print(f"Variance explained by first Principal Component: {explained_variance_ratio[0]:.2%}")
    print(f"Variance explained by first two Principal Components: {np.sum(explained_variance_ratio[:2]):.2%}")
    
    # Determine the biggest contributor to market risk (the first component)
    loadings = pd.DataFrame(pca.components_.T, columns=[f'PC{i+1}' for i in range(len(data.columns))], index=data.columns)
    risk_contributors = loadings['PC1'].abs().sort_values(ascending=False)
    
    print("\nContribution of each stock to the main market risk factor (PC1):")
    print(risk_contributors)
    
    # Visualize the Explained Variance
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, alpha=0.7, align='center', label='Individual explained variance')
    plt.step(range(1, len(explained_variance_ratio) + 1), np.cumsum(explained_variance_ratio), where='mid', label='Cumulative explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal component index')
    plt.title('PCA Explained Variance')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()
    
    return risk_contributors

def generate_recommendations(trends, risk):
    """
    Generates investment recommendations based on predicted trends and risk contributions.
    """
    print("\n--- Investment Recommendations ---")
    recommendations = {}
    
    # Identify the highest risk stock from PCA
    highest_risk_stock = risk.index[0]

    for ticker in trends.keys():
        recommendation = "Hold" # Default recommendation
        
        # Rule-based logic for recommendations
        if trends[ticker]['Trend'] == 'Up':
            # If predicted to go up, it's a potential buy
            recommendation = "Buy"
            # If it's also the highest risk contributor, be more cautious
            if ticker == highest_risk_stock:
                recommendation = "Hold (Volatile)"
        elif trends[ticker]['Trend'] == 'Down':
            # If predicted to go down, it's a potential sell
            recommendation = "Sell"
        
        recommendations[ticker] = recommendation
        print(f"{ticker}: {recommendation}")
        
    return recommendations

if __name__ == "__main__":
    # 1. Get data
    stock_data = fetch_data(TICKERS, START_DATE, END_DATE)
    
    # 2. Predict future trends using LSTM
    trend_predictions = predict_trends(stock_data)
    
    # 3. Analyze portfolio risk using PCA
    risk_analysis = perform_pca_risk_analysis(stock_data)
    
    # 4. Generate final investment recommendations
    final_recommendations = generate_recommendations(trend_predictions, risk_analysis)