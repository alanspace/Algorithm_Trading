import pandas as pd
import numpy as np
import yfinance as yf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# --- 1. Configuration and Game Plan ---
TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
NUM_STOCKS = len(TICKERS)
INITIAL_WEIGHTS = np.array([1/NUM_STOCKS] * NUM_STOCKS)

# --- THIS IS THE LINE TO CHANGE ---
# Set the cap HIGHER than the initial volatility of 0.3140
VOLATILITY_CAP = 0.35 # Changed from 0.25 to 0.35

# LSTM configuration
LOOKBACK_DAYS = 30
EPOCHS = 20

# --- 2. Fetching Stock Data ---
def fetch_stock_data(tickers, start_date, end_date):
    """
    Fetches adjusted closing prices for a list of tickers from Yahoo Finance.
    """
    print("Fetching stock data from Yahoo Finance...")
    stock_data = yf.download(tickers, start=start_date, end=end_date)
    # Use the 'Close' price which yfinance now provides as adjusted
    adj_close_prices = stock_data['Close'].dropna()
    print("Data fetched successfully.")
    return adj_close_prices

# --- 3. LSTM Data Preparation ---
def prepare_lstm_data(data, lookback_period):
    """
    Takes historical return data and prepares it for the LSTM model.
    Creates sequences of 'lookback_period' days to predict the next day's returns.
    """
    print("Preparing data for LSTM model...")
    X, y = [], []
    for i in range(lookback_period, len(data)):
        X.append(data.iloc[i-lookback_period:i].values)
        y.append(data.iloc[i].values)
    print(f"Created {len(X)} sequences of {lookback_period}-day lookbacks.")
    return np.array(X), np.array(y)

# --- 4. Building and Training the LSTM Model ---
def create_and_train_lstm(X_train, y_train):
    """
    Creates and trains the LSTM model with two LSTM layers and a Dense output layer.
    """
    print("Building the LSTM model...")
    model = Sequential([
        # Layer 1: LSTM with 50 units, returns sequences for the next LSTM layer
        LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        # Layer 2: LSTM with 50 units
        LSTM(50, return_sequences=False),
        # Output Layer: Dense layer with one neuron for each stock's prediction
        Dense(NUM_STOCKS)
    ])
    
    # Compile the model with Adam optimizer and Mean Squared Error for regression
    model.compile(optimizer='adam', loss='mean_squared_error')
    print("Model compiled. Starting training...")
    
    # Train the model for 20 epochs
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=32, verbose=1)
    print("Model training complete.")
    return model

# --- 5. Dynamically Adjusting Portfolio Weights ---
def adjust_portfolio_weights(predicted_returns, cov_matrix, volatility_cap):
    """
    Optimizes portfolio weights to maximize the Sharpe Ratio
    while keeping volatility below the defined cap.
    """
    num_assets = len(predicted_returns)
    
    # Objective function: We want to MINIMIZE the NEGATIVE Sharpe Ratio
    def objective(weights):
        # Expected portfolio return (annualized)
        portfolio_return = np.sum(predicted_returns * weights) * 252
        # Expected portfolio volatility (annualized)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
        
        # Penalty for exceeding volatility cap
        if portfolio_volatility > volatility_cap:
            return 1e6 # Return a large number to penalize this solution
        
        # Sharpe Ratio (assuming risk-free rate is 0 for simplicity)
        sharpe_ratio = portfolio_return / portfolio_volatility
        
        # Return negative Sharpe Ratio for minimization
        return -sharpe_ratio

    # Constraints: All weights must sum to 1
    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
    
    # Bounds: Each weight must be between 0 and 1 (no short selling)
    bounds = tuple((0, 1) for _ in range(num_assets))
    
    # Initial guess for weights
    initial_weights = np.array([1./num_assets] * num_assets)
    
    print("\nOptimizing portfolio weights to maximize Sharpe Ratio with volatility constraint...")
    # Perform the optimization
    result = minimize(objective, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    
    if not result.success:
        print("Optimization failed:", result.message)
        return initial_weights # Return initial weights on failure

    print("Optimization successful.")
    return result.x

# --- Main Execution ---
if __name__ == "__main__":
    # Define the time period (1 year from today)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    # 1. Fetch data
    price_data = fetch_stock_data(TICKERS, start_date, end_date)
    
    # 2. Calculate daily returns
    daily_returns = price_data.pct_change().dropna()
    
    # 3. Calculate initial portfolio volatility
    initial_cov_matrix = daily_returns.cov() * 252 # Annualize
    initial_volatility = np.sqrt(np.dot(INITIAL_WEIGHTS.T, np.dot(initial_cov_matrix, INITIAL_WEIGHTS)))
    print(f"\nInitial Portfolio Volatility (Equal Weights): {initial_volatility:.4f}")

    # 4. Prepare data for the LSTM
    X, y = prepare_lstm_data(daily_returns, LOOKBACK_DAYS)
    
    # 5. Create and train the model
    lstm_model = create_and_train_lstm(X, y)
    
    # 6. Predict the next day's risk/return trend
    # Use the last 'lookback_days' of data to predict the next day
    last_sequence = daily_returns[-LOOKBACK_DAYS:].values.reshape(1, LOOKBACK_DAYS, NUM_STOCKS)
    predicted_daily_returns = lstm_model.predict(last_sequence)[0]
    predicted_portfolio_return = np.sum(predicted_daily_returns * INITIAL_WEIGHTS)
    print(f"\nLSTM Predicted Next-Day Trend (Portfolio Return): {predicted_portfolio_return:.6f}")

    # 7. Dynamically adjust weights based on prediction
    # Use the historical covariance matrix as a measure of risk relationships
    historical_cov_matrix = daily_returns.cov()
    adjusted_weights = adjust_portfolio_weights(predicted_daily_returns, historical_cov_matrix, VOLATILITY_CAP)
    
    # 8. Display Results
    print("\n--- Final Results ---")
    print("Adjusted Portfolio Weights:")
    final_weights_df = pd.DataFrame({'Ticker': TICKERS, 'Weight': adjusted_weights})
    print(final_weights_df)
    print(f"\nSum of Adjusted Weights: {np.sum(adjusted_weights):.2f}")

    # 9. Visualize the optimized portfolio
    plt.figure(figsize=(10, 6))
    plt.bar(final_weights_df['Ticker'], final_weights_df['Weight'], color='skyblue')
    plt.title('Dynamically Optimized Portfolio Weights')
    plt.ylabel('Portfolio Weight')
    plt.xlabel('Stock Ticker')
    for i, weight in enumerate(adjusted_weights):
        plt.text(i, weight + 0.005, f"{weight:.2%}", ha='center')
    plt.show()