import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

def true_range(df):
    return pd.concat([
        df['High'] - df['Low'],
        abs(df['High'] - df['Close'].shift(1)),
        abs(df['Low'] - df['Close'].shift(1))
    ], axis=1).max(axis=1)

def lazybear_squeeze_momentum(df, length=20, mult=2.0, lengthKC=20, multKC=1.5, use_true_range=True):
    source = df['Close']

    # Bollinger Bands
    basis = source.rolling(length).mean()
    dev = mult * source.rolling(length).std()
    upperBB = basis + dev
    lowerBB = basis - dev

    # Keltner Channels
    if use_true_range:
        tr1 = df['High'] - df['Low']
        tr2 = abs(df['High'] - df['Close'].shift(1))
        tr3 = abs(df['Low'] - df['Close'].shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    else:
        tr = df['High'] - df['Low']

    rangema = tr.rolling(lengthKC).mean()
    ma = source.rolling(lengthKC).mean()
    upperKC = ma + multKC * rangema
    lowerKC = ma - multKC * rangema

    # Ensure all compared objects are Series with same index
    lowerBB, upperBB = lowerBB.align(df['Close'], join='right')
    lowerKC, upperKC = lowerKC.align(df['Close'], join='right')

    sqzOn = (lowerBB > lowerKC) & (upperBB < upperKC)
    sqzOff = (lowerBB < lowerKC) & (upperBB > upperKC)
    noSqz = ~sqzOn & ~sqzOff

    # LazyBear value
    highKC = df['High'].rolling(lengthKC).max()
    lowKC = df['Low'].rolling(lengthKC).min()
    avgHL = (highKC + lowKC) / 2
    midline = avgHL.rolling(lengthKC).mean()
    val_input = source - midline

    # Linear regression equivalent
    def linreg(vals):
        x = np.arange(len(vals))
        slope, intercept = np.polyfit(x, vals, 1)
        return slope * len(vals) + intercept

    val = val_input.rolling(lengthKC).apply(linreg, raw=True)

    # Color bars
    bcolor = np.where(val > 0,
                      np.where(val > val.shift(1), 'lime', 'green'),
                      np.where(val < val.shift(1), 'red', 'maroon'))

    scolor = np.where(noSqz, 'blue', np.where(sqzOn, 'black', 'gray'))

    # Add to df
    df['val'] = val
    df['bcolor'] = bcolor
    df['scolor'] = scolor
    df['squeeze_on'] = sqzOn
    df['squeeze_off'] = sqzOff

    return df

    source = df['Close']

    # Bollinger Bands
    basis = source.rolling(length).mean()
    dev = mult * source.rolling(length).std()
    upperBB = basis + dev
    lowerBB = basis - dev

    # Keltner Channels
    if use_true_range:
        tr1 = df['High'] - df['Low']
        tr2 = abs(df['High'] - df['Close'].shift(1))
        tr3 = abs(df['Low'] - df['Close'].shift(1))
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        range_ma = true_range.rolling(lengthKC).mean()
    else:
        range_ma = (df['High'] - df['Low']).rolling(lengthKC).mean()

    ma = source.rolling(lengthKC).mean()
    upperKC = ma + multKC * range_ma
    lowerKC = ma - multKC * range_ma

    # These are all Series now — no mismatched DataFrame comparisons
    sqzOn = (lowerBB > lowerKC) & (upperBB < upperKC)
    sqzOff = (lowerBB < lowerKC) & (upperBB > upperKC)
    noSqz = ~sqzOn & ~sqzOff

    # LazyBear momentum logic
    highKC = df['High'].rolling(lengthKC).max()
    lowKC = df['Low'].rolling(lengthKC).min()
    avgHL = (highKC + lowKC) / 2
    midline = avgHL.rolling(lengthKC).mean()

    val_input = source - midline
    val = val_input.rolling(lengthKC).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] * len(x) + np.polyfit(range(len(x)), x, 1)[1],
        raw=True
    )

    # Color logic
    bcolor = np.where(val > 0,
                      np.where(val > val.shift(1), 'lime', 'green'),
                      np.where(val < val.shift(1), 'red', 'maroon'))

    scolor = np.where(noSqz, 'blue',
                      np.where(sqzOn, 'black', 'gray'))

    df['val'] = val
    df['bcolor'] = bcolor
    df['scolor'] = scolor
    df['squeeze_on'] = sqzOn
    df['squeeze_off'] = sqzOff

    return df


def plot_lazybear_squeeze(df, ticker):
    fig, ax = plt.subplots(figsize=(14, 6))

    for i in range(len(df)):
        ax.bar(df.index[i], df['val'][i], color=df['bcolor'][i], width=0.8)

    ax.axhline(0, color='gray', linestyle='--')
    ax.set_title(f"{ticker} - Squeeze Momentum [LazyBear]")
    plt.tight_layout()
    plt.show()
    st.pyplot(plt)

# Usage
ticker = "AAPL"
start = "2022-01-01"
end = "2024-12-31"

df = yf.download(ticker, start=start, end=end, auto_adjust=True)
df = lazybear_squeeze_momentum(df)
plot_lazybear_squeeze(df, ticker)
