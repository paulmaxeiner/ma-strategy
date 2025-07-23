import streamlit as st
import matplotlib.pyplot as plt
from strategy import run_strategy

st.title("Moving Average Crossover Strategy")

#sidebar
st.sidebar.header("Strategy Parameters")
ticker = st.sidebar.text_input("Ticker", value="AAPL")
start_date = st.sidebar.date_input("Start Date", value="2020-01-01")
end_date = st.sidebar.date_input("End Date", value="2024-07-01")
short_window = st.sidebar.number_input("Short Moving Average", value=20, min_value=1)
long_window = st.sidebar.number_input("Long Moving Average", value=50, min_value=2)

#strat
if st.button("Run Strategy"):
    data, total_return, sharpe = run_strategy(ticker, start_date, end_date, short_window, long_window)

    st.subheader("Strategy vs Market Performance")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data.index, data["cumulative_market"], label="Market Return")
    ax.plot(data.index, data["cumulative_strategy"], label="Strategy Return")
    ax.legend()
    st.pyplot(fig)

    st.subheader("Performance Metrics")
    st.write(f"**Total Return:** {total_return:.2%}")
    st.write(f"**Sharpe Ratio:** {sharpe:.2f}")
