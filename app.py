import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
import datetime as dt
from keras.models import load_model
import streamlit as st

# model is based on the close price, can also be built on high, low, or open price

#initiate a start and end time for our dataframe
start = dt.datetime(2010, 1, 1)
#end = dt.datetime(2019, 12, 31)
end = dt.datetime.today()

#import streamlit library to host the webapp
st.title('Stock Trend Prediction')

#take user input
userInput = st.text_input('Enter Stock Ticker', 'AAPL')
df = data.DataReader(userInput, 'stooq', start, end)

#we have to reverse the dataframe because stooq gives us the data in reverse
df = df[::-1]

#Describing data
st.subheader('Data from 2010 - present')
st.write(df.describe())

#visualization
st.subheader('Closing Price vs Time chart')
fig = plt.figure(figsize = (12,6))
plt.plot(df.Close)
st.pyplot(fig)