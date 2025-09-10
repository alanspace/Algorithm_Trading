import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def moving_average_crossover(data, short_window, long_window):
    """
    Implements the moving average crossover strategy.

    Args:
        data (pd.DataFrame): DataFrame with stock prices, must contain a 'Close' column.
        short_window (int): The short moving average window.
        long_window (int): The long moving average window.

    Returns:
        float: The net profit from the strategy.
    """
    signals = pd.DataFrame(index=data.index)
    signals['signal'] = 0.0

    # Create short simple moving average over the short window
    # CHANGED: 'Adj Close' to 'Close'
    signals['short_mavg'] = data['Close'].rolling(window=short_window, min_periods=1, center=False).mean()

    # Create long simple moving average over the long window
    # CHANGED: 'Adj Close' to 'Close'
    signals['long_mavg'] = data['Close'].rolling(window=long_window, min_periods=1, center=False).mean()

    # Create signals
    signals['signal'][short_window:] = np.where(signals['short_mavg'][short_window:] > signals['long_mavg'][short_window:], 1.0, 0.0)

    # Generate trading orders
    signals['positions'] = signals['signal'].diff()

    # Initialize portfolio
    initial_capital = float(100000.0)
    positions = pd.DataFrame(index=signals.index).fillna(0.0)
    
    # Buy 100 shares
    positions['AAPL'] = 100 * signals['positions']
    
    # CHANGED: 'Adj Close' to 'Close'
    portfolio = positions.multiply(data['Close'], axis=0)
    pos_diff = positions.diff()
    
    # CHANGED: 'Adj Close' to 'Close'
    signals['holdings'] = (positions.multiply(data['Close'], axis=0)).sum(axis=1)
    
    # CHANGED: 'Adj Close' to 'Close'
    signals['cash'] = initial_capital - (pos_diff.multiply(data['Close'], axis=0)).sum(axis=1).cumsum()
    signals['total'] = signals['cash'] + signals['holdings']
    signals['returns'] = signals['total'].pct_change()

    return signals['total'][-1]


def find_best_parameters(data, short_window_range, long_window_range):
    """
    Performs a grid search to find the best moving average crossover parameters.

    Args:
        data (pd.DataFrame): DataFrame with stock prices.
        short_window_range (range): The range of short window values to test.
        long_window_range (range): The range of long window values to test.

    Returns:
        tuple: A tuple containing the best short window, best long window, and the best profit.
    """
    best_profit = -np.inf
    best_short_window = 0
    best_long_window = 0

    for short_window in short_window_range:
        for long_window in long_window_range:
            if short_window >= long_window:
                continue
            profit = moving_average_crossover(data, short_window, long_window)
            if profit > best_profit:
                best_profit = profit
                best_short_window = short_window
                best_long_window = long_window

    return best_short_window, best_long_window, best_profit


# Fetch Apple stock data from Yahoo Finance
# auto_adjust is True by default now
aapl_data = yf.download('AAPL', start='2020-01-01', end='2025-01-01')

# Define the grid for the grid search
short_window_range = range(10, 101, 10)
long_window_range = range(20, 201, 10)

# Find the best parameters
best_short, best_long, best_profit = find_best_parameters(aapl_data, short_window_range, long_window_range)

print(f"The best short window is: {best_short}")
print(f"The best long window is: {best_long}")
print(f"The best profit is: ${best_profit:,.2f}")

# Generate signals with the best parameters for plotting
signals = pd.DataFrame(index=aapl_data.index)
signals['signal'] = 0.0
# CHANGED: 'Adj Close' to 'Close'
signals['short_mavg'] = aapl_data['Close'].rolling(window=best_short, min_periods=1, center=False).mean()
# CHANGED: 'Adj Close' to 'Close'
signals['long_mavg'] = aapl_data['Close'].rolling(window=best_long, min_periods=1, center=False).mean()
signals['signal'][best_short:] = np.where(signals['short_mavg'][best_short:] > signals['long_mavg'][best_short:], 1.0, 0.0)
signals['positions'] = signals['signal'].diff()

# Plot the results
fig = plt.figure(figsize=(12, 8))
ax1 = fig.add_subplot(111, ylabel='Price in $')

# Plot the closing price
# CHANGED: 'Adj Close' to 'Close'
aapl_data['Close'].plot(ax=ax1, color='black', lw=2., label='Close Price')

# Plot the short and long moving averages
signals[['short_mavg', 'long_mavg']].plot(ax=ax1, lw=2.)

# Plot the buy signals
ax1.plot(signals.loc[signals.positions == 1.0].index,
         signals.short_mavg[signals.positions == 1.0],
         '^', markersize=10, color='g', label='buy')

# Plot the sell signals
ax1.plot(signals.loc[signals.positions == -1.0].index,
         signals.short_mavg[signals.positions == -1.0],
         'v', markersize=10, color='r', label='sell')

plt.title('Apple (AAPL) Moving Average Crossover Strategy')
plt.legend()
plt.show()