import streamlit as st
from datetime import date

import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objects as go

start = "2015-01-01"
end = date.today().strftime("%Y-%m-%d")

st.title("Stock Prediction App")

stocks = ("AAPL", "GOOG", "MSFT", "TSLA")
selectedStocks = st.selectbox("Select dateset for prediction", stocks)

nYears = st.slider("Years of prediction:", 1, 4)
period = nYears * 365

@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, start, end)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text("Load data...")
data = load_data(selectedStocks)
data_load_state.text("Loading data...done!")

st.subheader('Raw data')
st.write(data.tail())

def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='stock_open'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='stock_close'))
    fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()