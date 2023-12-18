import json
import time
from sklearn.preprocessing import MinMaxScaler
import requests
import pandas as pd
import plotly
from datetime import datetime, timedelta


def btcprice():
    url = 'https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT'

    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            bitcoin_price = float(data['price'])
            return bitcoin_price
        else:
            return None
    except requests.RequestException as e:
        print("Request Error:", e)
        return None


def gettrades(trades):
    table_content = "<table id='newsTable'>"
    table_content += "<tr><th>Trade ID</th><th>Trade Type</th><th>Quantity</th><th>Price</th><th>Percentage Change</th><th>USD Dollars Profit</th><th>Action</th></tr>"

    for trade in trades:
        if trade.active:
            table_content += f"<tr><td>{trade.trade_id}</td><td id='tradetype_{trade.trade_id}'>{trade.trade_type}</td><td>{trade.quantity}</td><td id='bp_{trade.trade_id}'>{trade.price}</td>"
            table_content += f"<td><span class='percentage_change' data-tradeid='{trade.trade_id}'>0.00%</span></td>"
            table_content += f"<td><span class='usd_profit' data-tradeid='{trade.trade_id}'>$0.00</span></td>"
            table_content += f"<td><form method='post' action='/close'>"  
            table_content += f"<button onclick='showLoader()' type='submit' name='tradeid' value='{trade.trade_id}'>Close</button>"
            table_content += f"<input type='hidden' id='tradeid_{trade.trade_id}' name='tradeid' value='{trade.trade_id}'>" 
            table_content += f"</form></td></tr>"  

    table_content += "</table>"

    # JavaScript function for live updates
    live_updates_script = """
<script>
    // Assuming you're updating livePrice periodically
    setInterval(updateTable, 5000); // Update every 5 seconds (for example)
function updateTable() {
    // Get the live price
    const livePrice = parseFloat(document.getElementById('livePrice').textContent.split(': ')[1]);

    // Get all elements with class 'percentage_change' and 'usd_profit'
    const percentageChangeElements = document.getElementsByClassName('percentage_change');
    const usdProfitElements = document.getElementsByClassName('usd_profit');

    // Loop through each trade element
    for (let i = 0; i < percentageChangeElements.length; i++) {
        const tradeId = percentageChangeElements[i].getAttribute('data-tradeid');
        const tradeType = document.getElementById(`tradetype_${tradeId}`).textContent; // Get trade type
        
        // Update Percentage Change and USD Dollars Profit based on trade type
        const priceElement = document.getElementById(`bp_${tradeId}`);
        const currentPrice = parseFloat(priceElement.textContent);
        const quantity = parseFloat(priceElement.nextElementSibling.textContent);
        let percentageChange, usdProfit;

        if (tradeType === 'long') {
            percentageChange = ((livePrice - currentPrice) / currentPrice) * 100;
            usdProfit = (livePrice - currentPrice) * quantity;
        } else if (tradeType === 'short') {
            percentageChange = ((currentPrice - livePrice) / currentPrice) * 100;
            usdProfit = (currentPrice - livePrice) * quantity;
        }

        percentageChangeElements[i].textContent = percentageChange.toFixed(2) + '%';
        usdProfitElements[i].textContent = '$' + usdProfit.toFixed(2);
    }
}

</script>
    """

    return table_content + live_updates_script