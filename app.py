from flask import Flask, render_template, redirect, url_for, request
from flask_socketio import SocketIO
import requests
import pandas as pd
import json
from datetime import datetime, timedelta
import websockets
from threading import Lock
import asyncio
import plotly.graph_objects as go
from models.trades_manager import TradesManager  # Assuming 'TradesManager' is a custom model
from models.trades_manager import TradesManager
from routes.trade import btcprice, gettrades
from routes.websocket import websocket_urls,  config
from routes.news import fetchnews, analyze_sentiment, global_df
from services.data_processing import get_open_interest_statistics, get_market_price_data
from services.sentiment_analysis import analyze_sentiment
from services.printchart import printchart

config = {
    "pair": "BTCUSDT",
    "interval": '1h',
}

websocket_urls = ["wss://fstream.binance.com/ws/" + config["pair"].lower() + "@aggTrade", "wss://wss.phoenixnews.io"]

acc = TradesManager()

app = Flask(__name__)
app.config['SERVER_NAME'] = 'testdash-pr7f.onrender.com'  # Replace with your actual server name
app.config['APPLICATION_ROOT'] = '/'
app.config['SECRET_KEY'] = 'donsky!'
socketio = SocketIO(app, cors_allowed_origins='*')

thread = None
thread_lock = Lock()

# SocketIO event handlers
@socketio.on('connect')
def connect():
    global thread
    print('Client connected')
    global thread
    with thread_lock:
        if thread is None:
            thread = socketio.start_background_task(background_thread)
    print('Client connected')

@socketio.on('disconnect')
def disconnect():
    print('Client disconnected')


async def handle_websocket(uri):
    """Handles WebSocket connections."""
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

async def process_data(data):
    """Processes incoming WebSocket data."""
    if data != "pong":
        try:
            data = json.loads(data)
            socketio.emit('update', data)
            print(data['p'])
            price = data['p']
            if 'receivedAt' in data:
                socketio.emit('data', data)
                print("Reload Index!")
                socketio.emit('reload', namespace='/')
            await asyncio.sleep(10)
        except Exception as e:
            print(f"Error processing data: {e}")

async def main():
    """Manages multiple WebSocket connections."""
    for uri in websocket_urls:
        print("\n\n\n------------------------------------\n")
        print(f"[+] Waiting for websocket msg from {uri}...")
        await asyncio.gather(handle_websocket(websocket_urls[0]), handle_websocket(websocket_urls[1]))

def background_thread():
    """Runs the background thread for WebSocket connections."""
    asyncio.run(main())

@app.route('/tradeBTC', methods=['GET', 'POST'])
def tradeBTC():
    """Handles trading of Bitcoin."""
    print("activater")
    error = None        
    if 'amount' in request.form:
        try:
            amount = float(request.form['amount'])
            action = request.form['action']
            print(action)
            price = btcprice()
            # Process based on the action (Update or Delete)
            if action == 'long':
                aa = amount/price 
                if acc.open_trade("long", aa, price):
                    return redirect(url_for('index')) 
            elif action == 'short':
                aa = amount/price 
                if acc.open_trade("short", aa, price):
                    return redirect(url_for('index')) 
            else:
                return "Invalid action!"
            return redirect(url_for('index')) 
        except ValueError:
            return "Invalid amount! Please provide a valid number."
    else:
        return "Amount not provided!"
    return redirect(url_for('index')) 

from flask import request

@app.route('/close', methods=['GET', 'POST'])
def close():
    """Closes trade."""
    print("activater")
    error = None
    if 'tradeid' in request.args or 'tradeid' in request.form:
        trade_id = int(request.args.get('tradeid') or request.form.get('tradeid'))  # Assign to trade_id instead of tradeid
        if acc.close_trade_by_id(trade_id, btcprice()):
            print("success")  # Corrected spelling of 'success'
            return redirect(url_for('index'))   # Add a return statement for response handling
    else:
        print("Trade ID not provided!")  # Modified error message for clarity
        return "Trade ID not provided!"
    return redirect(url_for('index')) 

@app.route('/')
def index():
    """Renders the index page."""
    print("Loading Index..")
    return render_template('index.html',  table_html=fetchnews(), firstchart=printchart(), Atrades=gettrades(acc.trades), balance=acc.balance)

if __name__ == '__main__':
    socketio.run(app)
    
