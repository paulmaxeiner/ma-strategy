import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

def run_strategy(ticker, start_date, end_date, short_window, long_window):
    data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)
    
    if data.empty:
        return None, None, None

    data["price"] = data["Close"]

    data["SMA_short"] = data["price"].rolling(window=short_window).mean()
    data["SMA_long"] = data["price"].rolling(window=long_window).mean()
   
    data["signal"] = 0

    data.loc[data["SMA_short"] > data["SMA_long"], "signal"] = 1
    data.loc[data["SMA_short"] < data["SMA_long"], "signal"] = -1

    data["returns"] = data["price"].pct_change()
    data["strategy_returns"] = data["signal"].shift(1) * data["returns"]
    data["cumulative_market"] = (1 + data["returns"]).cumprod()
    data["cumulative_strategy"] = (1 + data["strategy_returns"]).cumprod()

    if data["cumulative_strategy"].dropna().empty:
        return None, None, None

    total_return = data["cumulative_strategy"].iloc[-1] - 1
    sharpe = data["strategy_returns"].mean() / data["strategy_returns"].std() * (252 ** 0.5)

    return data, total_return, sharpe

