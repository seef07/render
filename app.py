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

                # Emit a message to the client to trigger a reload
                socketio.emit('reload', namespace='/')

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
    # Extract only the tbody content
    tbody_start = table_html.find('<tbody>')
    tbody_end = table_html.find('</tbody>') + len('</tbody>')
    tbody_content = table_html[tbody_start:tbody_end]

    return tbody_content

def analyze_sentiment(text):
    sid = SentimentIntensityAnalyzer()
    sentiment_scores = sid.polarity_scores(text)
    return sentiment_scores['compound']

global_df = pd.DataFrame(columns=['receivedAt', 'score'])

styles = [
    {'selector': 'th', 'props': [('color', 'white')]},

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
    "timestamp": ["Dec 2 at 10:00 PM","Dec 2 at 11:00 PM","Dec 2 at 3:00 PM","Dec 2 at 4:00 PM","Dec 2 at 5:00 PM","Dec 2 at 6:00 PM","Dec 2 at 7:00 PM","Dec 2 at 8:00 PM","Dec 2 at 9:00 PM","Dec 3 at 10:00 AM","Dec 3 at 10:00 PM","Dec 3 at 11:00 AM","Dec 3 at 11:00 PM","Dec 3 at 12:00 AM","Dec 3 at 12:00 PM","Dec 3 at 1:00 AM","Dec 3 at 1:00 PM","Dec 3 at 2:00 AM","Dec 3 at 2:00 PM","Dec 3 at 3:00 AM","Dec 3 at 3:00 PM","Dec 3 at 4:00 AM","Dec 3 at 4:00 PM","Dec 3 at 5:00 AM","Dec 3 at 5:00 PM","Dec 3 at 6:00 AM","Dec 3 at 6:00 PM","Dec 3 at 7:00 AM","Dec 3 at 7:00 PM","Dec 3 at 8:00 AM","Dec 3 at 8:00 PM","Dec 3 at 9:00 AM","Dec 3 at 9:00 PM","Dec 4 at 10:00 AM","Dec 4 at 10:00 PM","Dec 4 at 11:00 AM","Dec 4 at 11:00 PM","Dec 4 at 12:00 AM","Dec 4 at 12:00 PM","Dec 4 at 1:00 AM","Dec 4 at 1:00 PM","Dec 4 at 2:00 AM","Dec 4 at 2:00 PM","Dec 4 at 3:00 AM","Dec 4 at 3:00 PM","Dec 4 at 4:00 AM","Dec 4 at 4:00 PM","Dec 4 at 5:00 AM","Dec 4 at 5:00 PM","Dec 4 at 6:00 AM","Dec 4 at 6:00 PM","Dec 4 at 7:00 AM","Dec 4 at 7:00 PM","Dec 4 at 8:00 AM","Dec 4 at 8:00 PM","Dec 4 at 9:00 AM","Dec 4 at 9:00 PM","Dec 5 at 10:00 AM","Dec 5 at 10:00 PM","Dec 5 at 11:00 AM","Dec 5 at 11:00 PM","Dec 5 at 12:00 AM","Dec 5 at 12:00 PM","Dec 5 at 1:00 AM","Dec 5 at 1:00 PM","Dec 5 at 2:00 AM","Dec 5 at 2:00 PM","Dec 5 at 3:00 AM","Dec 5 at 3:00 PM","Dec 5 at 4:00 AM","Dec 5 at 4:00 PM","Dec 5 at 5:00 AM","Dec 5 at 5:00 PM","Dec 5 at 6:00 AM","Dec 5 at 6:00 PM","Dec 5 at 7:00 AM","Dec 5 at 7:00 PM","Dec 5 at 8:00 AM","Dec 5 at 8:00 PM","Dec 5 at 9:00 AM","Dec 5 at 9:00 PM","Dec 6 at 10:00 AM","Dec 6 at 10:00 PM","Dec 6 at 11:00 AM","Dec 6 at 11:00 PM","Dec 6 at 12:00 AM","Dec 6 at 12:00 PM","Dec 6 at 1:00 AM","Dec 6 at 1:00 PM","Dec 6 at 2:00 AM","Dec 6 at 2:00 PM","Dec 6 at 3:00 AM","Dec 6 at 3:00 PM","Dec 6 at 4:00 AM","Dec 6 at 4:00 PM","Dec 6 at 5:00 AM","Dec 6 at 5:00 PM","Dec 6 at 6:00 AM","Dec 6 at 6:00 PM","Dec 6 at 7:00 AM","Dec 6 at 7:00 PM","Dec 6 at 8:00 AM","Dec 6 at 8:00 PM","Dec 6 at 9:00 AM","Dec 6 at 9:00 PM","Dec 7 at 10:00 AM","Dec 7 at 10:00 PM","Dec 7 at 11:00 AM","Dec 7 at 11:00 PM","Dec 7 at 12:00 AM","Dec 7 at 12:00 PM","Dec 7 at 1:00 AM","Dec 7 at 1:00 PM","Dec 7 at 2:00 AM","Dec 7 at 2:00 PM","Dec 7 at 3:00 AM","Dec 7 at 3:00 PM","Dec 7 at 4:00 AM","Dec 7 at 4:00 PM","Dec 7 at 5:00 AM","Dec 7 at 5:00 PM","Dec 7 at 6:00 AM","Dec 7 at 6:00 PM","Dec 7 at 7:00 AM","Dec 7 at 7:00 PM","Dec 7 at 8:00 AM","Dec 7 at 8:00 PM","Dec 7 at 9:00 AM","Dec 7 at 9:00 PM","Dec 8 at 10:00 AM","Dec 8 at 10:00 PM","Dec 8 at 11:00 AM","Dec 8 at 11:00 PM","Dec 8 at 12:00 AM","Dec 8 at 12:00 PM","Dec 8 at 1:00 AM","Dec 8 at 1:00 PM","Dec 8 at 2:00 AM","Dec 8 at 2:00 PM","Dec 8 at 3:00 AM","Dec 8 at 3:00 PM","Dec 8 at 4:00 AM","Dec 8 at 4:00 PM","Dec 8 at 5:00 AM","Dec 8 at 5:00 PM","Dec 8 at 6:00 AM","Dec 8 at 6:00 PM","Dec 8 at 7:00 AM","Dec 8 at 7:00 PM","Dec 8 at 8:00 AM","Dec 8 at 8:00 PM","Dec 8 at 9:00 AM","Dec 8 at 9:00 PM","Dec 9 at 10:00 AM","Dec 9 at 11:00 AM","Dec 9 at 12:00 AM","Dec 9 at 12:00 PM","Dec 9 at 1:00 AM","Dec 9 at 1:00 PM","Dec 9 at 2:00 AM","Dec 9 at 2:00 PM","Dec 9 at 3:00 AM","Dec 9 at 4:00 AM","Dec 9 at 5:00 AM","Dec 9 at 6:00 AM","Dec 9 at 7:00 AM","Dec 9 at 8:00 AM","Dec 9 at 9:00 AM"],  # Include all your timestamps
    "bitcoin buy": ["61","54","35","38","41","40","47","44","55","35","58","40","63","53","31","62","33","49","34","48","37","40","33","39","41","38","39","37","40","37","48","35","49","49","89","53","79","72","50","66","51","62","55","58","58","62","49","68","68","63","61","54","65","50","71","46","71","41","100","43","96","76","42","66","44","77","45","64","51","63","55","58","74","59","80","49","76","75","78","46","82","42","71","54","64","91","50","79","46","77","51","75","55","71","52","65","58","66","57","49","63","46","58","47","62","40","67","46","67","69","42","67","42","60","42","60","54","64","49","58","51","51","55","54","57","45","56","39","58","39","65","38","61","68","39","53","38","54","43","51","49","54","44","48","44","42","45","46","49","41","49","33","60","37","42","66","37","53","39","57","40","53","47","47","41","47","40","39"],
    "value1": ["61","68","51","49","55","56","57","64","62","44","66","48","73","66","48","63","42","59","44","55","51","57","42","56","50","60","55","52","56","51","55","45","56","62","80","54","74","59","56","69","55","70","56","67","60","77","53","78","58","76","63","60","67","64","76","58","75","59","94","53","93","70","48","85","50","65","48","73","57","71","65","75","62","62","90","61","93","58","82","57","99","63","70","60","86","87","52","77","56","86","60","78","64","81","65","84","65","71","76","72","73","58","79","63","74","54","88","60","79","79","51","68","53","65","57","68","69","74","70","77","67","65","73","58","61","55","74","53","79","56","90","54","78","77","57","73","57","77","57","74","71","81","70","72","67","72","75","66","72","55","82","58","100","56","53","84","54","84","69","72","75","67","80","71","68","74","65","65"],
    "sell bitcoin": ["22","26","31","32","27","31","37","22","39","27","36","30","50","36","35","41","20","35","28","39","28","49","37","49","27","63","23","56","47","25","37","33","36","32","53","32","40","42","52","33","33","36","28","39","26","50","31","56","31","63","33","59","60","26","54","26","42","26","45","22","38","26","43","43","35","38","43","38","20","42","29","44","48","95","65","77","60","55","76","55","66","56","93","60","59","63","88","71","75","68","60","64","51","85","47","70","34","100","34","70","40","66","37","63","41","78","52","62","46","44","71","31","100","35","89","36","86","36","96","34","25","42","26","49","37","22","54","30","45","26","39","37","46","35","31","32","45","32","32","37","25","32","27","31","19","35","20","33","29","41","59","25","49","34","32","35","47","31","35","26","30","34","35","36","24","27","23","44"],
    "crypto sell": ["56","64","37","33","37","37","45","60","55","41","57","38","70","70","25","47","32","54","50","44","48","48","46","35","43","50","41","36","51","53","40","44","48","62","88","49","82","60","51","77","56","61","43","63","53","58","36","83","52","82","57","69","54","61","56","49","62","42","76","40","100","72","43","79","25","52","49","65","52","50","50","52","60","22","87","52","83","55","76","35","89","42","62","51","69","100","52","58","48","85","51","79","58","72","74","67","66","49","64","42","67","56","59","59","63","45","86","49","70","65","42","58","37","57","51","58","44","57","69","60","58","42","35","65","64","39","66","46","65","52","99","53","94","64","59","72","42","69","57","69","59","74","59","51","65","43","59","51","59","55","83","48","80","44","47","92","48","81","57","71","78","49","62","57","45","63","50","62"],
    "bitcoin": ["51","51","34","36","35","38","50","51","54","38","57","37","63","48","38","44","39","41","38","38","39","35","41","34","42","35","41","38","44","39","44","37","45","70","81","79","77","58","71","59","68","59","70","54","69","55","65","69","69","71","66","68","73","66","76","69","77","53","100","53","91","69","54","60","52","60","57","55","60","53","63","55","86","57","94","54","91","52","91","52","94","65","66","64","66","80","64","67","69","67","66","60","66","59","66","60","64","61","70","59","67","60","70","63","70","55","60","55","53","59","52","50","52","50","54","48","57","54","58","51","58","53","57","52","61","52","61","51","63","46","69","45","62","55","51","50","49","46","50","43","56","44","53","46","54","46","52","49","57","46","61","44","76","47","47","60","45","51","46","46","43","48","43","44","45","49","50","50"],
    "bitcoin sell": ["53","47","28","27","27","27","36","44","46","27","49","28","46","44","25","50","24","40","22","35","27","36","20","28","36","33","34","33","30","19","30","30","38","44","75","49","69","55","48","63","46","64","46","66","56","45","53","91","62","61","57","50","56","43","75","45","64","34","93","34","100","70","42","63","35","62","40","64","47","59","44","49","73","31","96","41","82","34","75","36","89","45","56","38","46","79","50","81","49","94","36","64","38","72","38","82","44","58","55","53","52","51","56","51","37","33","59","24","66","56","36","59","35","43","40","46","37","55","40","49","47","44","37","50","43","36","46","40","51","33","51","26","43","45","27","40","27","42","50","43","34","52","38","32","39","36","33","32","45","36","33","23","51","35","23","44","26","44","29","30","25","28","42","31","26","17","34","21"],
    "crypto": ["72","68","54","53","54","57","65","67","71","60","74","58","75","67","58","64","53","63","52","55","53","54","54","57","57","58","60","59","66","63","64","61","65","82","91","76","81","70","68","70","69","70","69","72","70","72","66","85","69","96","74","87","78","80","83","88","90","67","100","67","96","80","65","73","60","72","65","72","67","72","72","71","86","75","94","74","96","69","91","74","97","78","84","74","81","94","72","85","73","85","73","88","73","84","74","81","80","80","80","81","77","76","83","83","83","69","96","68","85","75","69","69","66","69","71","71","73","74","78","76","79","76","79","74","76","72","81","73","88","75","88","73","85","82","76","71","75","72","76","80","84","77","80","77","78","79","76","80","77","76","83","79","93","70","69","82","64","74","66","73","66","69","70","73","76","82","82","73"]
,
    }
data["value1"] = pd.to_numeric(data["value1"], errors="coerce")
data["bitcoin buy"] = pd.to_numeric(data["bitcoin buy"], errors="coerce")
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
    df_filtered = df[(df['source'] != 'Twitter')  & (df['body'] != '')].copy()

    return df_filtered

####################################################################################################################

# Serve root index file
@app.route('/')
def index():
    print("Loading Index..")
    return render_template('index.html',  table_html = fetchnews(), firstchart = printchart())
    
if __name__ == '__main__':
    socketio.run(app)
    