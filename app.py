import streamlit as st
import matplotlib.pyplot as plt
from strategy import run_strategy
from strategy import calculate_rsi
from strategy import squeeze_momentum
from strategy import plot_squeeze
import sm

st.title("Moving Average Crossover (with RSI) Strategy")

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

    combined_data = {
        "Market Return": data["cumulative_market"],
        "Strategy Return": data["cumulative_strategy"]
    }

    st.metric(label="Return", value=f"{total_return:.2%}", delta=f"{-1*(data["cumulative_market"].iloc[-1] - 1- total_return):.2%}")

    st.subheader("Performance Metrics")
    st.write(f"**Total Return:** {total_return:.2%}")
    st.write(f"**Sharpe Ratio:** {sharpe:.2f}")

    st.area_chart(combined_data, use_container_width=True)
   
    st.subheader("Strategy vs Market Performance")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data.index, data["cumulative_market"], label="Market Return")
    ax.plot(data.index, data["cumulative_strategy"], label="Strategy Return")
    ax.legend()
    st.pyplot(fig)

    sm.lazybear_squeeze_momentum(data)
    sm.plot_lazybear_squeeze(data, ticker)
    




    
    

