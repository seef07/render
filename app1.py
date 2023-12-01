from flask import Flask, render_template, request
from flask_socketio import SocketIO
from threading import Lock
import asyncio
import websockets
import json
import time
import requests
import pandas as pd
import plotly.graph_objects as go

# Background Thread
thread = None
thread_lock = Lock()

app = Flask(__name__)
app.config['SECRET_KEY'] = 'donsky!'
socketio = SocketIO(app, cors_allowed_origins='*')
config = {
    "pair": "BTCUSDT",
    "interval": '1h',
}

# Get current date time
def get_current_datetime():
    now = time.time()
    return now

# Generate random sequence of dummy sensor values and send it to our clients
def background_thread():
    print("WebSocket data thread")
    # Explicitly create and set up an event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    # Run the main function in the event loop
    loop.run_until_complete(main())

async def generate_candlestick_chart():
    # Define the API endpoint for Binance candlestick data
    api_url = 'https://api.binance.com/api/v3/klines'
    
    # Define the symbol and interval
    symbol = config['pair']
    interval = config['interval']
    
    # Specify additional parameters
    params = {
        'symbol': symbol,
        'interval': interval
    }

    try:
        response = requests.get(api_url, params=params)
        response.raise_for_status()  # Raise an exception for 4xx and 5xx status codes
        data = response.json()
        print("Candlestick Data Sent to Client:")
        print(data)
        # Emit the data to the frontend
        socketio.emit('updateCandlestickChart', {'data': data})
    except requests.RequestException as e:
        print(f"Failed to fetch candlestick data: {e}")
        # Emit an error message to the frontend
        socketio.emit('updateCandlestickChart', {'error': str(e)})

# ...



# Serve root index file
@app.route('/')
def index():
    return render_template('index.html')

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
                    await asyncio.sleep(5)

        except websockets.ConnectionClosed:
            print("WebSocket connection closed. Reconnecting...")
            await asyncio.sleep(5)

        except Exception as e:
            print(f"An error occurred: {e}")
            await asyncio.sleep(5)

# Process the incoming data
async def process_data(data):
    if data != "pong":
        try:
            data = json.loads(data)
            socketio.emit('updateSensorData', data)
            print("Sensor Data Sent to Client:")
            await generate_candlestick_chart()
            print(data)

        except Exception as e:
            print(f"Error processing data: {e}")

@app.route('/submit', methods=['POST'])
def submit():
    pair = request.form.get('pair')
    interval = request.form.get('interval')
    config['pair'] = pair
    config['interval'] = interval
    # Process the form data as needed
    return render_template('index.html')

# Main function
async def main():
    uri = f'wss://fstream.binance.com/ws/{config["pair"].lower()}@aggTrade'  
    print("\n\n\n------------------------------------\n")
    print("[+] Waiting for websocket msg...")
    print("hey")
    await handle_websocket(uri)

if __name__ == '__main__':
    socketio.run(app)
