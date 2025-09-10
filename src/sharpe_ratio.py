import pandas as pd
import numpy as np
import yfinance as yf
from scipy.optimize import minimize

# --- Configuration ---
STOCKS = ['AAPL', 'TSLA', 'AMZN', 'GOOG', 'META']
NUM_STOCKS = len(STOCKS)
START_DATE = '2018-01-01'
END_DATE = pd.to_datetime('today').strftime('%Y-%m-%d')
RISK_FREE_RATE = 0.02

def get_daily_data(symbols, start, end):
    """
    Fetches daily adjusted closing prices from Yahoo Finance.
    """
    print("Fetching daily adjusted data from Yahoo Finance...")
    try:
        # yfinance downloads data for all tickers at once.
        # The 'Close' column is used because auto_adjust=True is the default.
        stock_data = yf.download(symbols, start=start, end=end)
        
        # --- THIS IS THE CORRECTED LINE ---
        # Select the 'Close' column, which now holds the adjusted price.
        price_data = stock_data['Close'].dropna()
        
        print(" > Successfully fetched all data.")
        return price_data
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

def calculate_portfolio_performance(weights, mean_returns, cov_matrix):
    """
    Calculates the expected annual return, volatility, and Sharpe Ratio for a portfolio.
    """
    portfolio_return = np.sum(mean_returns * weights) * 252
    portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    sharpe_ratio = (portfolio_return - RISK_FREE_RATE) / portfolio_std_dev
    return portfolio_return, portfolio_std_dev, sharpe_ratio

def negative_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
    """
    The function to be minimized.
    """
    return -calculate_portfolio_performance(weights, mean_returns, cov_matrix)[2]

def optimize_portfolio(returns):
    """
    Finds the optimal portfolio weights that maximize the Sharpe Ratio.
    """
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(NUM_STOCKS))
    initial_weights = np.array([1./NUM_STOCKS] * NUM_STOCKS)

    print("\nOptimizing portfolio for maximum Sharpe Ratio...")
    result = minimize(fun=negative_sharpe_ratio,
                      x0=initial_weights,
                      args=(mean_returns, cov_matrix, RISK_FREE_RATE),
                      method='SLSQP',
                      bounds=bounds,
                      constraints=constraints)
    
    return result.x

# --- Main Execution ---
if __name__ == "__main__":
    price_data = get_daily_data(STOCKS, START_DATE, END_DATE)

    if price_data is not None and not price_data.empty:
        log_returns = np.log(price_data / price_data.shift(1)).dropna()

        print("\n--- Initial Portfolio (Equal Weights) ---")
        initial_weights = np.array([1./NUM_STOCKS] * NUM_STOCKS)
        for i, stock in enumerate(STOCKS):
            print(f"Initial weight for {stock}: {initial_weights[i]:.2%}")
        
        initial_return, initial_vol, initial_sharpe = calculate_portfolio_performance(
            initial_weights, log_returns.mean(), log_returns.cov()
        )
        print(f"\nExpected Annual Return: {initial_return:.2%}")
        print(f"Annual Volatility: {initial_vol:.2%}")
        print(f"Sharpe Ratio: {initial_sharpe:.2f}")

        optimal_weights = optimize_portfolio(log_returns)
        print("\n--- Optimized Portfolio (Max Sharpe Ratio) ---")
        for i, stock in enumerate(STOCKS):
            print(f"Optimal weight for {stock}: {optimal_weights[i]:.2%}")

        opt_return, opt_vol, opt_sharpe = calculate_portfolio_performance(
            optimal_weights, log_returns.mean(), log_returns.cov()
        )
        print(f"\nExpected Annual Return: {opt_return:.2%}")
        print(f"Annual Volatility: {opt_vol:.2%}")
        print(f"Sharpe Ratio: {opt_sharpe:.2f}")