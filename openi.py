import requests
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
import plotly


def get_open_interest_statistics(symbol, period='5m', limit=500):
    base_url = "https://fapi.binance.com"
    endpoint = "/futures/data/openInterestHist"
    params = {
        "symbol": symbol,
        "period": period,
        "limit": limit,
    }

    url = f"{base_url}{endpoint}"
    response = requests.get(url, params=params)
    data = response.json()
    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')  # Convert timestamp to datetime
    return df
# Add a function to get candlestick data
def get_market_price_data(symbol, interval='5m', limit=500):
    base_url = 'https://api.binance.com/api/v3/klines'
    
    params = {
        'symbol': symbol,
        'interval': interval,
        'limit': limit,
    }

    response = requests.get(base_url, params=params)
    data = response.json()
    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')  # Convert timestamp to datetime
    return df[['timestamp', 'close']]  # Return both 'timestamp' and 'close' columns



# Define the symbols for the pairs
symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT']

# Create a MinMaxScaler
scaler = MinMaxScaler()
# ... (previous code)
fig = go.Figure()
# Plot data for each symbol
for symbol in symbols:
    # Get data from the first API (Binance)
    binance_data = get_open_interest_statistics(symbol, period='5m', limit=500)

    # Get data from the second API (replace with actual API details)
    second_api_data = get_market_price_data(symbol, interval='5m', limit=500)
    print(second_api_data)
    print(binance_data)
    # Merge or concatenate the datasets based on a common column, e.g., 'timestamp'
    # Assuming both datasets have a 'timestamp' column
    combined_data = pd.merge(binance_data, second_api_data, on='timestamp', how='outer')

    # Normalize the numerical columns using Min-Max scaling
    numerical_columns = ['sumOpenInterest', 'close']  # Replace with actual column names
    combined_data[numerical_columns] = scaler.fit_transform(combined_data[numerical_columns])
    if symbol == "BTCUSDT":
        fig.add_trace(go.Scatter(x=combined_data['timestamp'], y=combined_data['close'], mode='lines', name=f'{symbol} - Market price', line=dict(color='black', width = 2)))
    
    fig.add_trace(go.Scatter(x=combined_data['timestamp'], y=combined_data['sumOpenInterest'], mode='lines', name=f'{symbol} - Open Interest Data'))
    

fig.update_layout(title='Open Interest Data Comparison',
                  xaxis_title='Timestamp',
                  yaxis_title='Normalized Value',
                  legend_title='Symbols')

fig.show()
