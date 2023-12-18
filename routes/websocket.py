config = {
    "pair": "BTCUSDT",
    "interval": '1h',
}
websocket_urls = ["wss://fstream.binance.com/ws/" + config["pair"].lower() + "@aggTrade", "wss://wss.phoenixnews.io"]
