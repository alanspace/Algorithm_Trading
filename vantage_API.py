import pandas as pd
from alpha_vantage.timeseries import TimeSeries
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("API_KEY")

if api_key is None:
    print("Error: API_KEY not found in .env or environment variables.")
else:
    print(f"API Key: {api_key}")

STOCK_SYMBOL = 'AAPL'

def get_real_time_apple_data(api_key):
    """
    Fetches real-time intraday stock data for Apple (AAPL) from Alpha Vantage
    and returns it as a pandas DataFrame.

    Args:
        api_key (str): Your Alpha Vantage API key.

    Returns:
        pandas.DataFrame: A DataFrame containing the intraday stock data,
                          or None if an error occurs.
    """
    try:
        # 1. Initialize the TimeSeries class with your API key
        ts = TimeSeries(key=api_key, output_format='pandas')

        # 2. Fetch intraday data for the specified stock symbol
        # The get_intraday function returns the data and metadata.
        # Interval can be '1min', '5min', '15min', '30min', or '60min'.
        data, meta_data = ts.get_intraday(symbol=STOCK_SYMBOL, interval='5min', outputsize='compact')

        # 3. Rename columns for clarity
        # The default column names from the API are like '1. open', '2. high', etc.
        data.rename(columns={
            '1. open': 'Open',
            '2. high': 'High',
            '3. low': 'Low',
            '4. close': 'Close',
            '5. volume': 'Volume'
        }, inplace=True)
        
        # The index is already a datetime object, which is convenient
        data.index.name = 'Timestamp'

        return data

    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please check the following:")
        print("1. Your API key is correct and has not expired.")
        print("2. You have not exceeded the API call limit (5 calls per minute for a free key).")
        print("3. The stock symbol is correct.")
        return None

# --- Main Execution ---
if __name__ == "__main__":
    if api_key == 'API_KEY':
        print("="*60)
        print("WARNING: Please replace 'API_KEY' with your actual key.")
        print("You can get a free key from: https://www.alphavantage.co/support/#api-key")
        print("="*60)
    else:
        # Get the real-time data
        apple_df = get_real_time_apple_data(api_key)

        # Print the resulting DataFrame
        if apple_df is not None:
            print(f"Successfully fetched real-time data for {STOCK_SYMBOL}")
            print("="*40)
            print("Latest Data Points (5-minute interval):")
            print(apple_df.head()) # Print the most recent data points
            print("\nDataFrame Info:")
            apple_df.info()