import streamlit as st
from datetime import date

import yfinance as yfin
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objects as go

def valid_ticker(t):
    """
    Check whether given ticker is a valid stock symbol.

    NOTE: Assumes that a stock is valid IF Yahoo! Finance returns info for a ticker

    Args:
        ticker (str): Ticker symbol in question.

    Returns:
        if an error have been raised, an error message will be returned
        else return the ticker and the price 
    """
    tickers = yfin.Ticker(t)

    try:
        price = round(get_current_price(t), 2)
        return(f"Current Price of {t}: {price:.2f}")
    except:
        return(f"Cannot get info of {t}, it probably does not exist")

def get_current_price(t):
    """
    get the last closing price of a stock t

    NOTE: Assumes that a stock is valid because this function is passed through the valid_ticker function

    Args:
        ticker (str): Ticker symbol in question.

    Returns:
        return the last closed priced of a stock
    """
    ticker = yfin.Ticker(t)
    todays_data = ticker.history(period='1d')
    return todays_data['Close'][0]

start = "2010-01-01"
end = date.today().strftime("%Y-%m-%d")

st.title("Stock Prediction App")

# stocks = ("AAPL", "GOOG", "MSFT", "TSLA")
# selectedStocks = st.selectbox("Select dateset for prediction", stocks)


#take user input
st.write("Enter Stock Ticker Below")
userInput = st.text_input('Example: AAPL, GOOG, MSFT, TSLA', 'AAPL')

#validates user input
st.write(valid_ticker(userInput))

nYears = st.slider("Years of prediction:", 1, 4)
period = nYears * 365

@st.cache_data
def load_data(ticker):
    data = yfin.download(ticker, start, end)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text("Load data...")
data = load_data(userInput)
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


# forecasting
df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

model = Prophet()
model.fit(df_train)
future = model.make_future_dataframe(periods=period)
forecast = model.predict(future)

st.subheader('Forecast data')
st.write(forecast.tail())

st.write('forecast data')
fig1 = plot_plotly(model, forecast)
st.plotly_chart(fig1)

st.write('forecast components')
fig2 = model.plot_components(forecast)
st.write(fig2)


