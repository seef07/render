import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import asyncio
import websockets
import json
import time
import matplotlib.pyplot as plt
from flask import Flask, render_template
from flask_socketio import SocketIO
from threading import Lock
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
import requests
import pandas as pd
import plotly
from datetime import datetime, timedelta


plt.style.use('fivethirtyeight')

analyzer = SentimentIntensityAnalyzer()
config = {
    "pair": "BTCUSDT",
    "interval": '1h',
}
# Background Thread
thread = None
thread_lock = Lock()

app = Flask(__name__)
app.config['SECRET_KEY'] = 'donsky!'
socketio = SocketIO(app, cors_allowed_origins='*')

# List of WebSocket URLs you want to connect to
websocket_urls = ["wss://fstream.binance.com/ws/" + config["pair"].lower() + "@aggTrade", "wss://wss.phoenixnews.io"]

# Decorator for connect
@socketio.on('connect')
def connect():
    global thread
    print('Client connected')
    global thread
    with thread_lock:
        if thread is None:
            thread = socketio.start_background_task(background_thread)

# Decorator for disconnect
@socketio.on('disconnect')
def disconnect():
    print('Client disconnected')

# Websocket listener
async def handle_websocket(uri):
    while True:
        try:
            async with websockets.connect(uri) as websocket:
                while True:
                    data = await websocket.recv()
                    await process_data(data)

        except websockets.ConnectionClosed:
            print(f"WebSocket connection closed for {uri}. Reconnecting...")
            await asyncio.sleep(5)

        except Exception as e:
            print(f"An error occurred for {uri}: {e}")
            await asyncio.sleep(5)

# Process the incoming data
async def process_data(data):
    if data != "pong":
        try:
            data = json.loads(data)
            socketio.emit('update', data)
            print("Data send :)")
            if 'receivedAt' in data:
                socketio.emit('data', data)
                return render_template('index.html',  table_html = fetchnews(),firstchart = printchart())
                print("Success!")
            await asyncio.sleep(10)

        except Exception as e:
            print(f"Error processing data: {e}")

# Main function
async def main():
    for uri in websocket_urls:
        print("\n\n\n------------------------------------\n")
        print(f"[+] Waiting for websocket msg from {uri}...")
        await asyncio.gather(handle_websocket(websocket_urls[0]), handle_websocket(websocket_urls[1]))


# Background Thread
def background_thread():
    asyncio.run(main())

##########################################################################################################

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

def printchart():
    symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT']
    # Create a MinMaxScaler
    scaler = MinMaxScaler()
    # ... (previous code)
    fig = go.Figure()
    # Assuming global_df['score'] is a numeric column




# Fit and transform the scores


# Assign the scaled scores back to the DataFrame

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
            fig.add_trace(go.Scatter(x=combined_data['timestamp'] + pd.Timedelta(hours=1), y=combined_data['close'], mode='lines', name=f'{symbol} - Market price', line=dict(color='black', width = 2)))
            print("News")
        scores = global_df['score'].values.reshape(-1, 1)
        scaled_scores = scaler.fit_transform(scores)
        global_df['scaled_score'] = scaled_scores.flatten()
        fig.add_trace(go.Scatter(x=global_df['receivedAt'], y=global_df['scaled_score'], mode='markers', marker=dict(color=global_df['score'], colorscale='RdYlGn', size=4), name='Sentiment Score', showlegend=False))

        fig.add_trace(go.Scatter(x=combined_data['timestamp']+ pd.Timedelta(hours=1), y=combined_data['sumOpenInterest'], mode='lines', name=f'{symbol} - Open Interest Data'))
        

    fig.update_layout(title='Open Interest Data Comparison',
                    xaxis_title='Timestamp',
                    yaxis_title='Normalized Value',
                    legend_title='Symbols')

    grapp  = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)



    return grapp
###################################################################### 
def fetchnews():
    url =  'https://api.phoenixnews.io/getLastNews?limit=200'
    response = requests.get(url)
    data = response.json()
    df = pd.DataFrame(data)    
    # List of columns to remove
    df['Text'] = df['body'].fillna('') + ' ' + df['description'].fillna('')

    columns_to_remove = ['_id','body','image3','image4', 'imageQuote2', 'imageQuote3', 'imageQuote4','image', 'description', 'createdAt', 'url', 'title', 'suggestions', 'category', 'isReply', 'coin', 'image1', 'username', 'name', 'icon', 'twitterId', 'tweetId', 'isRetweet', 'isQuote', 'image', 'imageQuote', 'image2']

    # Drop specified columns
    df = df.drop(columns=columns_to_remove, errors='ignore')
    df['score'] = df['Text'].apply(analyze_sentiment)
    df['receivedAt'] = df['receivedAt'].apply(lambda x: datetime.strptime(x, '%Y-%m-%dT%H:%M:%S.%fZ').strftime('%Y-%m-%d %H:%M:%S'))
    df['receivedAt'] = pd.to_datetime(df['receivedAt'])

    print(df)

    global_df['receivedAt'] = df['receivedAt']
    global_df['receivedAt'] = global_df['receivedAt'] + timedelta(hours=1)
    global_df['score'] = df['score']
    print(global_df)
    
    table_html = df.style.set_table_styles(styles).render(classes='table', index=False)
    return table_html

def analyze_sentiment(text):
    sid = SentimentIntensityAnalyzer()
    sentiment_scores = sid.polarity_scores(text)
    return sentiment_scores['compound']

global_df = pd.DataFrame(columns=['receivedAt', 'score'])

styles = [
    {'selector': 'th', 'props': [('background-color', '#4CAF50'), ('color', 'white'), ('border', '2px solid black')]},
    {'selector': 'td', 'props': [('border', '1px solid #dddddd'), ('text-align', 'left'), ('padding', '8px')]},
    {'selector': 'tr:nth-child(even)', 'props': [('background-color', '#f2f2f2')]},
    {'selector': 'table', 'props': [('border-collapse', 'separate'), ('border-spacing', '0'), ('width', '70%'), ('margin', 'auto')]},
    {'selector': 'td, th', 'props': [('border-radius', '5px')]},
    {'selector': 'table', 'props': [('border', '1px solid black'), ('width', '70%'), ('margin', 'auto')]},
]
#######################################################################
from datetime import datetime

# Initialize an empty list
timestamp_value_list = []

# Function to add timestamp and value to the list
def add_timestamp_value(timestamp, value):
    timestamp_value_list.append({'timestamp': timestamp, 'value': value})

# Example usage



# Serve root index file
@app.route('/')
def index():
    print("ghye")
    print(global_df)
    return render_template('index.html',  table_html = fetchnews(), firstchart = printchart())
    
if __name__ == '__main__':
    socketio.run(app)
    