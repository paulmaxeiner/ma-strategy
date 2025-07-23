import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

def squeeze_momentum(df, bb_window=20, kc_window=20, mult_bb=2, mult_kc=1.5):
    df['20sma'] = df['Close'].rolling(bb_window).mean()
    df['20std'] = df['Close'].rolling(bb_window).std()
    df['upper_bb'] = df['20sma'] + mult_bb * df['20std']
    df['lower_bb'] = df['20sma'] - mult_bb * df['20std']

    df['TR'] = df[['High', 'Low', 'Close']].max(axis=1) - df[['High', 'Low', 'Close']].min(axis=1)
    df['ATR'] = df['TR'].rolling(kc_window).mean()
    df['upper_kc'] = df['20sma'] + mult_kc * df['ATR']
    df['lower_kc'] = df['20sma'] - mult_kc * df['ATR']

    df['squeeze_on'] = (df['lower_bb'] > df['lower_kc']) & (df['upper_bb'] < df['upper_kc'])
    df['momentum'] = df['Close'] - df['Close'].shift(4)

    # Buy signal: squeeze just ended and momentum rising
    df['squeeze_off'] = (~df['squeeze_on']) & (df['squeeze_on'].shift(1))
    df['buy_signal'] = (df['squeeze_off']) & (df['momentum'] > 0)

    # Sell signal: momentum just turned negative
    df['sell_signal'] = (df['momentum'] < 0) & (df['momentum'].shift(1) > 0)
    print(df)
    return df

def plot_squeeze(df, ticker):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df.index, df['momentum'], label='Momentum', color='purple')
    ax.fill_between(df.index, 0, df['momentum'], where=df['momentum'] > 0, color='green', alpha=0.4)
    ax.fill_between(df.index, 0, df['momentum'], where=df['momentum'] < 0, color='red', alpha=0.4)
    
    ax.scatter(df.index[df['squeeze_on']], df['momentum'][df['squeeze_on']],
               color='black', label='Squeeze On', s=30)

    ax.scatter(df.index[df['buy_signal']], df['momentum'][df['buy_signal']],
               color='blue', label='Buy Signal', s=60, marker='^')

    ax.scatter(df.index[df['sell_signal']], df['momentum'][df['sell_signal']],
               color='orange', label='Sell Signal', s=60, marker='v')

    ax.axhline(0, color='gray', linestyle='--', linewidth=1)
    ax.set_title(f'{ticker} - Squeeze Momentum with Buy/Sell Signals')
    ax.legend()
    st.pyplot(fig)

    

def calculate_rsi(data, period=14):
    delta = data['Close'].diff()

    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()

    for i in range(period, len(data)):
        avg_gain.iloc[i] = ((avg_gain.iloc[i-1] * (period - 1)) + gain.iloc[i]) / period
        avg_loss.iloc[i] = ((avg_loss.iloc[i-1] * (period - 1)) + loss.iloc[i]) / period

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi

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

    smi = squeeze_momentum(data)

    data["buy_signal"] = smi["buy_signal"]
    data["sell_signal"] = smi["sell_signal"]

    data.loc[data["buy_signal"], "signal"] = 1
    data.loc[data["sell_signal"], "signal"] = -1

    data["rsi"] = calculate_rsi(data)

    data.loc[data["rsi"] > 80, "signal"] = -1
    data.loc[data["rsi"] < 20, "signal"] = 1

    data["returns"] = data["price"].pct_change()
    data["strategy_returns"] = data["signal"].shift(1) * data["returns"]
    data["cumulative_market"] = (1 + data["returns"]).cumprod()
    data["cumulative_strategy"] = (1 + data["strategy_returns"]).cumprod()
    

    if data["cumulative_strategy"].dropna().empty:
        return None, None, None

    total_return = data["cumulative_strategy"].iloc[-1] - 1
    sharpe = data["strategy_returns"].mean() / data["strategy_returns"].std() * (252 ** 0.5)
    plot_squeeze(squeeze_momentum(data),ticker)
    return data, total_return, sharpe

