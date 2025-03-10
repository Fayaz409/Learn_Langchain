import streamlit as st
import yfinance as yf
import plotly.graph_objects as go

# Title and description
st.title("Simple Trading App Demo")
st.write("This app downloads AAPL market data and displays a candlestick chart.")

# Define the stock symbol and download data
symbol = "AAPL"
data = yf.download(symbol, period="1y", progress=False)

if data.empty:
    st.error("No market data found for symbol: " + symbol)
else:
    # Create a candlestick chart
    fig = go.Figure(data=[go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name="Candlestick"
    )])
    
    fig.update_layout(title=f"{symbol} Candlestick Chart", xaxis_title="Date", yaxis_title="Price ($)")
    st.plotly_chart(fig, use_container_width=True)

# A simple button to simulate an agent run (you can later expand this)
if st.button("Run Simple Trade Simulation"):
    st.write("Running a simple simulation... (This is where you would run your agent logic.)")
    # Example: display the latest closing price
    latest_price = data['Close'].iloc[-1]
    st.write(f"Latest closing price for {symbol}: ${latest_price:.2f}")
