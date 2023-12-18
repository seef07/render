from flask import Flask, render_template, redirect, url_for, request
from flask_socketio import SocketIO
from threading import Lock
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime, timedelta
from services.data_processing import get_open_interest_statistics, get_market_price_data
from routes.news import fetchnews, analyze_sentiment, global_df
import csv
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import json
import plotly

def gettrend(start):
    file_path = 'trend.csv'

    with open(file_path, newline='', encoding='utf-8-sig') as csvfile:
        reader = csv.reader(csvfile)
        lines = list(reader)

        # Store the first line in the variable 'timestamp'
        timestamp = lines[0][:-1]
        crypto = lines[1][:-1]
        bitcoin = lines[2][:-1]
        cryptosell = lines[3][:-1]
        bitcoinsell = lines[4][:-1]
        bitcoinbuy = lines[5][:-1]
        cryptobuy = lines[6][:-1]

    # Construct the data dictionary with one-dimensional lists
    data = {
        "timestamp": timestamp,
        "crypto": crypto,
        "bitcoin": bitcoin,
        "crypto sell": cryptosell,
        "bitcoin sell": bitcoinsell,
        "bitcoin buy": bitcoinbuy,
        "crypto buy": cryptobuy
    }

    # Create a DataFrame
    df = pd.DataFrame(data)
 
    
    # Convert the timestamp column to datetime format
    df['timestamp'] = pd.to_datetime(df['timestamp'], format="%b %d at %I:%M %p")

    df['timestamp'] = df['timestamp'].map(lambda x: x.replace(year=datetime.now().year))
    columns_to_convert = ['bitcoin', 'crypto', 'bitcoin sell', 'crypto sell', 'bitcoin buy', 'crypto buy']
    df[columns_to_convert] = df[columns_to_convert].apply(pd.to_numeric, errors='coerce')

    # Set the timestamp as the index
    df.set_index('timestamp', inplace=True)
    # Resample the data to 5-minute intervals and interpolate the values
    # Resample the data to 5-minute intervals and interpolate the values
    df_resampled = df.resample('5T').interpolate()


    start_date = start

    df_fil = df_resampled[df_resampled.index >= start_date]
    print(df_fil)
    return df_fil

plt.style.use('fivethirtyeight')
def printchart(symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT']):
    # Create a MinMaxScaler
    scaler = MinMaxScaler()
    # ... (previous code)
    fig = go.Figure()
    # Assuming global_df['score'] is a numeric column
    # Plot data for each symbol
    for symbol in symbols:
        # Get data from the first API (Binance)
        binance_data = get_open_interest_statistics(symbol, period='5m', limit=500)

        # Get data from the second API (replace with actual API details)
        second_api_data = get_market_price_data(symbol, interval='5m', limit=500)
        print("marketdata api success")
        print("binance api success")
        # Merge or concatenate the datasets based on a common column, e.g., 'timestamp'
        # Assuming both datasets have a 'timestamp' column
        combined_data = pd.merge(binance_data, second_api_data, on='timestamp', how='outer')
        
        # Normalize the numerical columns using Min-Max scaling
        numerical_columns = ['sumOpenInterest', 'close']  # Replace with actual column names
        combined_data[numerical_columns] = scaler.fit_transform(combined_data[numerical_columns])
        if symbol == "BTCUSDT":
            fig.add_trace(go.Scatter(x=combined_data['timestamp'] + pd.Timedelta(hours=1), y=combined_data['close'], mode='lines', name=f'{symbol} - Market price', line=dict(color='white', width = 2)))
        scores = global_df['score'].values.reshape(-1, 1)
        scaled_scores = scaler.fit_transform(scores)
        global_df['scaled_score'] = scaled_scores.flatten()
        fig.add_trace(go.Scatter(x=global_df['receivedAt'], y=global_df['scaled_score'], mode='markers', marker=dict(color=global_df['score'], colorscale='RdYlGn', size=4), name='Sentiment Score', showlegend=False))

        fig.add_trace(go.Scatter(x=combined_data['timestamp']+ pd.Timedelta(hours=1), y=combined_data['sumOpenInterest'], mode='lines', name=f'{symbol} - Open Interest Data', visible='legendonly'))
    trend_data = gettrend(combined_data['timestamp'].min())


    # Normalize the numerical columns of trend_data
    trend_data[['crypto', 'bitcoin', 'crypto sell', 'bitcoin sell', 'bitcoin buy', 'crypto buy']] = scaler.fit_transform(trend_data[['crypto', 'bitcoin', 'crypto sell', 'bitcoin sell', 'bitcoin buy', 'crypto buy']])

    # Plot trend data
    fig.add_trace(go.Scatter(x=trend_data.index, y=trend_data['crypto'], mode='lines', name=f' Trend Data (crypto1)', line=dict(width=2), visible='legendonly'))
    fig.add_trace(go.Scatter(x=trend_data.index, y=trend_data['bitcoin'], mode='lines', name=f'Trend Data (Sell Bitcoin)', line=dict(width=2), visible='legendonly'))
    fig.add_trace(go.Scatter(x=trend_data.index, y=trend_data['crypto sell'], mode='lines', name=f'Trend Data (Crypto Sell)', line=dict(width=2), visible='legendonly'))
    fig.add_trace(go.Scatter(x=trend_data.index, y=trend_data['bitcoin sell'], mode='lines', name=f'Trend data (Bitcoin) sell', line=dict(width=2), visible='legendonly'))
    fig.add_trace(go.Scatter(x=trend_data.index, y=trend_data['bitcoin buy'], mode='lines', name=f'{symbol} - Trend Data (Bitcoin Sell)', line=dict(width=2), visible='legendonly'))
    fig.add_trace(go.Scatter(x=trend_data.index, y=trend_data['crypto buy'], mode='lines', name=f'Trend Data (Crypto buy)', line=dict(width=2), visible='legendonly'))



    fig.update_layout(title='Open Interest Data Comparison',
                    xaxis_title='Timestamp',
                    yaxis_title='Normalized Value',
                    legend_title='Symbols')
    fig.update_layout(template='plotly_dark')
    grapp  = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)



    return grapp