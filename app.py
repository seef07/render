import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import asyncio
import websockets
import json
import time
import matplotlib.pyplot as plt
from flask import Flask, render_template, redirect, url_for

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
app.config['SERVER_NAME'] = '127.0.0.1:5000'  # Replace with your actual server name
app.config['APPLICATION_ROOT'] = '/'
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
                print("Reload Index!")

                # Use flask's app context to call render_template
                with app.app_context():
                    print("Inside app context")
                    return redirect(url_for('index'))
                
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
    trend_data[['value1', 'sell bitcoin', 'crypto sell', 'bitcoin', 'bitcoin sell', 'crypto']] = scaler.fit_transform(trend_data[['value1', 'sell bitcoin', 'crypto sell', 'bitcoin', 'bitcoin sell', 'crypto']])

    # Plot trend data
    fig.add_trace(go.Scatter(x=trend_data.index, y=trend_data['value1'], mode='lines', name=f' Trend Data (Value 1)', line=dict(width=2), visible='legendonly'))
    fig.add_trace(go.Scatter(x=trend_data.index, y=trend_data['sell bitcoin'], mode='lines', name=f'rend Data (Sell Bitcoin)', line=dict(width=2), visible='legendonly'))
    fig.add_trace(go.Scatter(x=trend_data.index, y=trend_data['crypto sell'], mode='lines', name=f'Trend Data (Crypto Sell)', line=dict(width=2), visible='legendonly'))
    fig.add_trace(go.Scatter(x=trend_data.index, y=trend_data['bitcoin'], mode='lines', name=f' Trend Data (Bitcoin)', line=dict(width=2), visible='legendonly'))
    fig.add_trace(go.Scatter(x=trend_data.index, y=trend_data['bitcoin sell'], mode='lines', name=f'{symbol} - Trend Data (Bitcoin Sell)', line=dict(width=2), visible='legendonly'))
    fig.add_trace(go.Scatter(x=trend_data.index, y=trend_data['crypto'], mode='lines', name=f'Trend Data (Crypto)', line=dict(width=2), visible='legendonly'))



    fig.update_layout(title='Open Interest Data Comparison',
                    xaxis_title='Timestamp',
                    yaxis_title='Normalized Value',
                    legend_title='Symbols')
    fig.update_layout(template='plotly_dark')
    grapp  = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)



    return grapp
###################################################################### 
def fetchnews():
    url =  'https://api.phoenixnews.io/getLastNews?limit=200'
    response = requests.get(url)
    data = response.json()
    df = filtertje(pd.DataFrame(data))    
    # List of columns to remove
    df['Text'] = df['body'].fillna('') + ' ' + df['description'].fillna('')

    columns_to_remove = ['_id','body','image3','image4', 'imageQuote2', 'imageQuote3', 'imageQuote4','image', 'description', 'createdAt', 'url', 'title', 'suggestions', 'category', 'isReply', 'coin', 'image1', 'username', 'name', 'icon', 'twitterId', 'tweetId', 'isRetweet', 'isQuote', 'image', 'imageQuote', 'image2', "important"]

    # Drop specified columns
    df = df.drop(columns=columns_to_remove, errors='ignore')
    df['score'] = df['Text'].apply(analyze_sentiment)
    df['receivedAt'] = df['receivedAt'].apply(lambda x: datetime.strptime(x, '%Y-%m-%dT%H:%M:%S.%fZ').strftime('%Y-%m-%d %H:%M:%S'))
    df['receivedAt'] = pd.to_datetime(df['receivedAt'])


    global_df['receivedAt'] = df['receivedAt']
    global_df['receivedAt'] = global_df['receivedAt'] + timedelta(hours=1)
    global_df['score'] = df['score']
    
    table_html = df.style.set_table_styles(styles).render(classes='table', index=False)
    return table_html

def analyze_sentiment(text):
    sid = SentimentIntensityAnalyzer()
    sentiment_scores = sid.polarity_scores(text)
    return sentiment_scores['compound']

global_df = pd.DataFrame(columns=['receivedAt', 'score'])

styles = [
    {'selector': 'th', 'props': [('background-color', '##3A4D4D'), ('color', 'white'), ('border', '2px solid white')]},
    {'selector': 'td', 'props': [('border', '1px solid white'), ('text-align', 'left'), ('padding', '8px')]},
    {'selector': 'tr:nth-child(even)', 'props': [('background-color', '#0D0D0D')]},
    {'selector': 'table', 'props': [('border-collapse', 'separate'), ('border-spacing', '0'), ('width', '70%'), ('margin', 'auto')]},
    {'selector': 'td, th', 'props': [('border-radius', '5px')]},
    {'selector': 'table', 'props': [('border', '1px solid white'), ('width', '70%'), ('margin', 'auto')]},
]
#######################################################################
from datetime import datetime

# Initialize an empty list
timestamp_value_list = []

# Function to add timestamp and value to the list
def add_timestamp_value(timestamp, value):
    timestamp_value_list.append({'timestamp': timestamp, 'value': value})

# Example usage

#############################################gettrend.py##########################################################################################
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Your data (timestamps and values)
data = {
    "timestamp": ["Dec 1 at 10:00 AM","Dec 1 at 10:00 PM","Dec 1 at 11:00 AM","Dec 1 at 11:00 PM","Dec 1 at 12:00 AM","Dec 1 at 12:00 PM","Dec 1 at 1:00 AM","Dec 1 at 1:00 PM","Dec 1 at 2:00 AM","Dec 1 at 2:00 PM","Dec 1 at 3:00 AM","Dec 1 at 3:00 PM","Dec 1 at 4:00 AM","Dec 1 at 4:00 PM","Dec 1 at 5:00 AM","Dec 1 at 5:00 PM","Dec 1 at 6:00 AM","Dec 1 at 6:00 PM","Dec 1 at 7:00 AM","Dec 1 at 7:00 PM","Dec 1 at 8:00 AM","Dec 1 at 8:00 PM","Dec 1 at 9:00 AM","Dec 1 at 9:00 PM","Dec 2 at 10:00 AM","Dec 2 at 10:00 PM","Dec 2 at 11:00 AM","Dec 2 at 11:00 PM","Dec 2 at 12:00 AM","Dec 2 at 12:00 PM","Dec 2 at 1:00 AM","Dec 2 at 1:00 PM","Dec 2 at 2:00 AM","Dec 2 at 2:00 PM","Dec 2 at 3:00 AM","Dec 2 at 3:00 PM","Dec 2 at 4:00 AM","Dec 2 at 4:00 PM","Dec 2 at 5:00 AM","Dec 2 at 5:00 PM","Dec 2 at 6:00 AM","Dec 2 at 6:00 PM","Dec 2 at 7:00 AM","Dec 2 at 7:00 PM","Dec 2 at 8:00 AM","Dec 2 at 8:00 PM","Dec 2 at 9:00 AM","Dec 2 at 9:00 PM","Dec 3 at 10:00 AM","Dec 3 at 10:00 PM","Dec 3 at 11:00 AM","Dec 3 at 11:00 PM","Dec 3 at 12:00 AM","Dec 3 at 12:00 PM","Dec 3 at 1:00 AM","Dec 3 at 1:00 PM","Dec 3 at 2:00 AM","Dec 3 at 2:00 PM","Dec 3 at 3:00 AM","Dec 3 at 3:00 PM","Dec 3 at 4:00 AM","Dec 3 at 4:00 PM","Dec 3 at 5:00 AM","Dec 3 at 5:00 PM","Dec 3 at 6:00 AM","Dec 3 at 6:00 PM","Dec 3 at 7:00 AM","Dec 3 at 7:00 PM","Dec 3 at 8:00 AM","Dec 3 at 8:00 PM","Dec 3 at 9:00 AM","Dec 3 at 9:00 PM","Dec 4 at 10:00 AM","Dec 4 at 10:00 PM","Dec 4 at 11:00 AM","Dec 4 at 11:00 PM","Dec 4 at 12:00 AM","Dec 4 at 12:00 PM","Dec 4 at 1:00 AM","Dec 4 at 1:00 PM","Dec 4 at 2:00 AM","Dec 4 at 2:00 PM","Dec 4 at 3:00 AM","Dec 4 at 3:00 PM","Dec 4 at 4:00 AM","Dec 4 at 4:00 PM","Dec 4 at 5:00 AM","Dec 4 at 5:00 PM","Dec 4 at 6:00 AM","Dec 4 at 6:00 PM","Dec 4 at 7:00 AM","Dec 4 at 7:00 PM","Dec 4 at 8:00 AM","Dec 4 at 8:00 PM","Dec 4 at 9:00 AM","Dec 4 at 9:00 PM","Dec 5 at 10:00 AM","Dec 5 at 10:00 PM","Dec 5 at 11:00 AM","Dec 5 at 12:00 AM","Dec 5 at 12:00 PM","Dec 5 at 1:00 AM","Dec 5 at 1:00 PM","Dec 5 at 2:00 AM","Dec 5 at 2:00 PM","Dec 5 at 3:00 AM","Dec 5 at 3:00 PM","Dec 5 at 4:00 AM","Dec 5 at 4:00 PM","Dec 5 at 5:00 AM","Dec 5 at 5:00 PM","Dec 5 at 6:00 AM","Dec 5 at 6:00 PM","Dec 5 at 7:00 AM","Dec 5 at 7:00 PM","Dec 5 at 8:00 AM","Dec 5 at 8:00 PM","Dec 5 at 9:00 AM","Dec 5 at 9:00 PM","Nov 28 at 11:00 PM","Nov 29 at 10:00 AM","Nov 29 at 10:00 PM","Nov 29 at 11:00 AM","Nov 29 at 11:00 PM","Nov 29 at 12:00 AM","Nov 29 at 12:00 PM","Nov 29 at 1:00 AM","Nov 29 at 1:00 PM","Nov 29 at 2:00 AM","Nov 29 at 2:00 PM","Nov 29 at 3:00 AM","Nov 29 at 3:00 PM","Nov 29 at 4:00 AM","Nov 29 at 4:00 PM","Nov 29 at 5:00 AM","Nov 29 at 5:00 PM","Nov 29 at 6:00 AM","Nov 29 at 6:00 PM","Nov 29 at 7:00 AM","Nov 29 at 7:00 PM","Nov 29 at 8:00 AM","Nov 29 at 8:00 PM","Nov 29 at 9:00 AM","Nov 29 at 9:00 PM","Nov 30 at 10:00 AM","Nov 30 at 10:00 PM","Nov 30 at 11:00 AM","Nov 30 at 11:00 PM","Nov 30 at 12:00 AM","Nov 30 at 12:00 PM","Nov 30 at 1:00 AM","Nov 30 at 1:00 PM","Nov 30 at 2:00 AM","Nov 30 at 2:00 PM","Nov 30 at 3:00 AM","Nov 30 at 3:00 PM","Nov 30 at 4:00 AM","Nov 30 at 4:00 PM","Nov 30 at 5:00 AM","Nov 30 at 5:00 PM","Nov 30 at 6:00 AM","Nov 30 at 6:00 PM","Nov 30 at 7:00 AM","Nov 30 at 7:00 PM","Nov 30 at 8:00 AM","Nov 30 at 8:00 PM","Nov 30 at 9:00 AM","Nov 30 at 9:00 PM"],  # Include all your timestamps
    "bitcoin buy": [35, 58, 36, 56, 45, 30, 50, 33, 48, 32, 46, 34, 44, 37, 45, 39, 41, 42, 39, 43, 37, 45, 33, 51, 34, 59, 33, 52, 48, 31, 53, 36, 57, 36, 57, 33, 48, 39, 44, 42, 47, 42, 38, 46, 39, 44, 38, 54, 36, 57, 40, 64, 51, 28, 63, 33, 50, 33, 47, 36, 40, 32, 39, 41, 38, 37, 35, 39, 38, 47, 34, 49, 49, 88, 53, 77, 69, 49, 63, 52, 62, 57, 54, 59, 65, 46, 71, 67, 59, 62, 54, 66, 50, 70, 41, 70, 35, 100, 38, 64, 37, 59, 41, 67, 38, 55, 47, 55, 52, 53, 67, 50, 71, 44, 73, 78, 69, 39, 75, 53, 30, 47, 37, 56, 53, 32, 48, 33, 52, 37, 46, 41, 46, 35, 43, 40, 36, 44, 39, 39, 35, 41, 27, 44, 36, 50, 31, 56, 55, 32, 49, 31, 43, 37, 43, 38, 43, 47, 42, 39, 36, 41, 36, 47, 35, 42, 33, 50],
    "value1": ["48","66","49","76","59","47","62","46","65","48","60","42","63","57","61","52","65","55","52","64","52","57","52","65","56","71","51","76","65","51","63","57","67","56","68","59","56","54","74","63","73","63","69","64","59","75","52","69","54","75","54","80","74","55","72","48","66","50","57","58","64","49","61","59","67","62","59","64","57","59","50","66","72","89","62","88","68","61","78","62","77","65","73","69","85","61","88","64","85","68","74","77","74","87","70","84","59","92","52","67","50","84","50","72","47","75","56","73","66","72","65","66","90","60","89","63","85","59","100","61","51","68","48","66","73","49","60","43","56","49","65","50","56","46","60","56","54","59","56","58","54","64","46","64","51","73","47","71","59","47","61","43","52","50","60","51","62","68","63","56","57","56","56","58","46","70","50","71"],
    "sell bitcoin": ["22","26","31","32","27","31","37","22","39","27","36","30","50","36","35","41","20","35","28","39","28","49","37","49","27","63","23","56","47","25","37","33","36","32","53","32","40","42","52","33","33","36","28","39","26","50","31","56","31","63","33","59","60","26","54","26","42","26","45","22","38","26","43","43","35","38","43","38","20","42","29","44","48","95","65","77","60","55","76","55","66","56","93","60","59","63","88","71","75","68","60","64","51","85","47","70","34","100","34","70","40","66","37","63","41","78","52","62","46","44","71","31","100","35","89","36","86","36","96","34","25","42","26","49","37","22","54","30","45","26","39","37","46","35","31","32","45","32","32","37","25","32","27","31","19","35","20","33","29","41","59","25","49","34","32","35","47","31","35","26","30","34","35","36","24","27","23","44"],
    "crypto sell": ["53","53","37","57","54","39","50","35","64","49","46","36","61","39","53","53","61","58","56","52","25","43","32","56","43","74","43","68","65","39","48","57","57","49","59","40","60","46","43","45","42","43","35","51","42","75","47","70","50","69","41","77","79","28","49","35","64","53","63","59","51","47","40","52","63","50","44","63","61","61","49","58","91","100","67","88","76","60","88","72","75","44","67","56","66","48","100","62","92","73","76","69","72","70","64","78","42","77","49","79","41","73","29","58","49","68","53","62","56","55","67","34","84","57","85","59","74","38","90","40","41","52","39","68","63","45","64","50","63","34","45","45","51","46","39","38","36","45","49","45","51","46","55","46","39","63","30","67","38","36","51","35","63","51","40","43","55","37","41","56","42","54","40","51","31","52","32","42"],
    "bitcoin": ["42","49","39","46","37","39","36","36","32","37","33","39","33","41","33","45","35","42","35","44","35","48","36","50","36","51","35","51","43","35","39","37","38","37","37","35","35","35","33","35","35","38","36","50","37","51","36","54","38","58","38","62","43","38","45","38","39","38","39","39","35","42","35","41","35","41","36","45","40","45","38","45","70","80","76","76","57","68","57","69","58","68","57","70","57","65","69","67","70","69","71","72","68","77","68","79","54","100","51","67","53","60","54","57","54","53","62","55","63","55","83","55","96","55","92","55","89","50","90","43","34","40","35","38","40","37","36","37","34","36","32","37","32","39","32","37","31","38","33","38","34","38","35","41","31","41","33","39","37","35","36","33","33","36","32","37","31","38","31","37","31","37","29","36","31","40","31","41"],
    "bitcoin sell": ["20","24","28","29","24","28","33","19","35","24","32","27","44","32","31","37","18","32","26","35","25","44","33","44","24","57","20","50","42","23","33","29","33","29","47","28","36","37","46","29","29","32","25","35","24","45","28","50","28","56","29","52","54","24","48","23","38","23","41","20","34","23","39","39","31","34","38","34","18","37","26","39","43","85","58","69","53","49","68","49","59","50","83","54","53","57","79","63","67","61","54","58","46","76","43","63","30","100","30","62","36","59","33","57","37","70","47","55","41","40","64","28","89","31","80","32","77","32","86","31","23","38","23","44","33","20","49","27","41","23","35","33","41","31","27","29","40","29","29","33","22","29","25","28","17","31","18","30","26","37","53","22","44","30","29","32","42","28","31","23","27","31","31","33","21","24","21","40"],
    "crypto": ["59","75","59","73","64","59","58","56","57","58","64","52","63","60","64","63","67","62","66","66","63","69","65","73","65","79","61","74","64","58","63","58","63","59","58","58","62","60","65","60","68","64","65","72","66","73","63","77","66","83","65","84","74","60","68","58","66","55","59","56","58","59","58","60","61","62","67","67","65","68","68","72","89","100","84","90","75","76","73","77","80","75","76","76","78","74","95","76","96","81","92","87","92","91","92","99","66","100","66","75","65","70","63","72","67","72","69","71","71","74","86","75","93","69","98","70","95","68","96","69","64","72","63","66","63","62","61","62","60","64","64","62","61","58","61","64","64","67","67","62","67","68","68","75","60","69","60","66","59","59","59","57","63","58","67","59","63","70","64","64","65","61","65","64","62","70","61","69"],
    }
data["value1"] = pd.to_numeric(data["value1"], errors="coerce")
data["sell bitcoin"] = pd.to_numeric(data["sell bitcoin"], errors="coerce")
data["crypto sell"] = pd.to_numeric(data["crypto sell"], errors="coerce")
data["bitcoin"] = pd.to_numeric(data["bitcoin"], errors="coerce")
data["bitcoin sell"] = pd.to_numeric(data["bitcoin sell"], errors="coerce")
data["crypto"] = pd.to_numeric(data["crypto"], errors="coerce")

######################### ^DATA Preprocessing^ ##############################################

def gettrend(start):
    # Create a DataFrame
    df = pd.DataFrame(data)


# Convert the timestamp column to datetime format
    df['timestamp'] = pd.to_datetime(df['timestamp'], format="%b %d at %I:%M %p")

    df['timestamp'] = df['timestamp'].map(lambda x: x.replace(year=datetime.now().year))

    print(start)
# Set the timestamp as the index
    df.set_index('timestamp', inplace=True)
# Filter data based on start and end dates

# Resample the data to 5-minute intervals and interpolate the values
    df_resampled = df.resample('5T').interpolate()

    start_date = start
    end_date = "2023-11-30 20:55:00"

    df_fil = df_resampled[(df_resampled.index >= start_date)]

    return(df_fil)

#############################################################################################################################################################
# filter


def filtertje(df):
    # Drop rows where 'Source' is equal to 'Twitter'
    # Assuming 'source' column is in lowercase, adjust accordingly if needed
    df_filtered = df[(df['source'] != 'twitter')  & (df['body'] != '')].copy()

    return df_filtered

####################################################################################################################

# Serve root index file
@app.route('/')
def index():
    print("Loading Index..")
    return render_template('index.html',  table_html = fetchnews(), firstchart = printchart())
    
if __name__ == '__main__':
    socketio.run(app)
    