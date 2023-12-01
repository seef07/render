import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import math
def time_to_expiration(expiration_date):
    # Convert the expiration date string to a datetime object
    expiration_datetime = datetime.strptime(expiration_date, '%Y-%m-%d')

    # Get the current date and time
    current_datetime = datetime.now()

    # Calculate the time to expiration in days
    time_to_expiration_days = (expiration_datetime - current_datetime).days

    # Convert days to years (assuming 252 trading days in a year)
    time_to_expiration_years = time_to_expiration_days / 252.0

    return time_to_expiration_years

# Example usage
expiration_date = '2023-12-08'
time_to_exp = time_to_expiration(expiration_date)

def options_chain(symbol):

    tk = yf.Ticker(symbol)
    # Expiration dates
    exps = tk.options

    # Get options for each expiration
    options = pd.DataFrame()
    for e in exps:
        opt = tk.option_chain(e)
        opt = pd.DataFrame().append(opt.calls).append(opt.puts)
        opt['expirationDate'] = e
        options = options.append(opt, ignore_index=True)

    # Bizarre error in yfinance that gives the wrong expiration date
    # Add 1 day to get the correct expiration date
    options['expirationDate'] = pd.to_datetime(options['expirationDate']) + datetime.timedelta(days = 1)
    options['dte'] = (options['expirationDate'] - datetime.datetime.today()).dt.days / 365
    
    # Boolean column if the option is a CALL
    options['CALL'] = options['contractSymbol'].str[4:].apply(
        lambda x: "C" in x)
    
    options[['bid', 'ask', 'strike']] = options[['bid', 'ask', 'strike']].apply(pd.to_numeric)
    options['mark'] = (options['bid'] + options['ask']) / 2 # Calculate the midpoint of the bid-ask
    
    # Drop unnecessary and meaningless columns
    options = options.drop(columns = ['contractSize', 'currency', 'change', 'percentChange', 'lastTradeDate', 'lastPrice'])

    return options


def black_scholes_call(S, X, T, r, sigma):
    d1 = (math.log(S / X) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    
    N_d1 = 0.5 * (1 + math.erf(d1 / math.sqrt(2)))
    N_d2 = 0.5 * (1 + math.erf(d2 / math.sqrt(2)))
    
    call_price = S * N_d1 - X * math.exp(-r * T) * N_d2
    
    return call_price

def get_data(symbol, exp):
    tk = yf.Ticker(symbol)
    exps = tk.option_chain(exp)
    call_columns_to_print = ['contractSymbol', 'strike', 'impliedVolatility', 'bid', 'openInterest', 'ask']
    return exps[0][call_columns_to_print]

def get_chain(symbol):
    tk = yf.Ticker(symbol)
    chain = tk.options
    return chain

def get_cprice(symbol):
    tk = yf.Ticker(symbol)
    current_price = tk.info['currentPrice']
    return current_price


if __name__ == "__main__":
    symbol = "BLNK"
    exp = get_chain(symbol)[1]
    r = 0.04416
###################################
    columns = ['ContractSymbol', 'ExpData', 'CurrentPrice', 'strike', 'BondsInterest', 'impliedVolatility', 'OpenInt', 'ask', 'BS', 'Bin']
    df = pd.DataFrame(columns=columns)
###################################
    frame = get_data(symbol, exp)
    print(frame.columns)
    ran = len(frame)
    T = time_to_expiration(exp)
    S = get_cprice(symbol)
####################################
    for i in range(0,ran):
        sym = frame.iloc[i]['contractSymbol']
        x = frame.iloc[i]['strike']
        sigma = frame.iloc[i]['impliedVolatility']
        bs = black_scholes_call(S, x, T, r, sigma)
        openInterest = frame.iloc[i]['openInterest']
        ask = frame.iloc[i]['ask']
        ab = 0
        if ask < bs:
            ab = 1
    	######################################
        data = {'ContractSymbol': sym, 'ExpData': T,'CurrentPrice': S, 'strike': x, 'BondsInterest': r, 'impliedVolatility': sigma, 'OpenInt': openInterest , 'ask': ask, 'BS': bs, 'Bin' : ab}
        df = df.append(data, ignore_index=True)
#    print(.iloc[1])
    print("################################")
    print()
#Index(['contractSymbol', 'lastTradeDate', 'strike', 'lastPrice', 'bid', 'ask', 'change', 'percentChange', 'volume', 'openInterest','impliedVolatility', 'inTheMoney', 'contractSize', 'currency'],
# risk free rate : 4.4160
print(df)


    # Time to expiration in years
 # Risk-free interest rate
