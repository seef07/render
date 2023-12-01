## same but with price change!

import requests
import numpy as np 
import plotly.graph_objects as go
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from datetime import datetime

market = 'BTCUSDT'
tick_interval = '5m'

url = 'https://api.binance.com/api/v3/klines?symbol='+market+'&interval='+tick_interval
data = requests.get(url).json()

timestamps = [datap[0] for datap in data[:1000]]
closeprice = [float(datap[4]) for datap in data[:1000]]
volume = [float(datap[5]) for datap in data[:1000]]
trades = [int(datap[8]) for datap in data[:1000]]
smartmoney = np.divide(volume, trades)

Nsmartmoney = (smartmoney - np.min(smartmoney)) / (np.max(smartmoney) - np.min(smartmoney))
Ncloseprice = (closeprice - np.min(closeprice)) / (np.max(closeprice) - np.min(closeprice))
closeprice_changes = np.diff(Ncloseprice)
closeprice_changes = np.insert(closeprice_changes, 0, np.nan)

timestampss = [datetime.utcfromtimestamp(timestamp / 1000).strftime('%Y-%m-%d %H:%M:%S') for timestamp in timestamps]

next_close_prices = closeprice[12:]

# Print lengths of arrays for debugging
# Check if the lengths are consistent
if len(timestampss) == len(Nsmartmoney) == len(Ncloseprice) == len(closeprice):
    print("Lengths are consistent.")

    # Create DataFrame
    df = pd.DataFrame(list(zip(timestampss, Nsmartmoney, Ncloseprice, closeprice_changes, ['Increase' if next_close_prices[i] > closeprice[i] else 'Decrease' for i in range(len(next_close_prices))])),
                      columns=['Timestamp', 'Normalized_SmartMoney', 'Normalized_ClosePrice', "closechange", 'MarketDirection'])

    # Drop the last row to align features and target variable
    df = df[:-12]

else:
    print("Lengths are not consistent.")

X = df[['Normalized_SmartMoney', 'Normalized_ClosePrice']]
y = df['MarketDirection']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a logistic regression model
model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Display classification report
print(classification_report(y_test, y_pred))



###### Creat a graph
fig = go.Figure()
fig.add_trace(go.Scatter(x=timestampss, y=Nsmartmoney, mode='lines', name='Normalized Smart Money'))
fig.add_trace(go.Scatter(x=timestampss, y=closeprice_changes, mode='lines', name='Normalized Close Price change'))
fig.add_trace(go.Scatter(x=timestampss, y=Ncloseprice, mode='lines', name='Normalized Close Price'))

fig.update_layout(
    title='Normalized Smart Money and Close Price Over Time',
    xaxis_title='Timestamp',
    yaxis_title='Normalized Value',
)

fig.show()