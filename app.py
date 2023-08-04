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
plt.plot(df.Close, label="Closing Price")
plt.legend(loc="upper left")
plt.xlabel("Year")
plt.ylabel("Price")
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 100MA')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100, label="ma100")
plt.plot(df.Close, label="Closing Price")
plt.legend(loc="upper left")
plt.xlabel("Year")
plt.ylabel("Price")
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 100MA and 200MA')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100, label="ma100")
plt.plot(ma200, label="ma200")
plt.plot(df.Close, label="Closing Price")
plt.legend(loc="upper left")
plt.xlabel("Year")
plt.ylabel("Price")
st.pyplot(fig)

#split the data into Training and Testing
#train 70% of the data, and test the rest
data_training = pd.DataFrame(df['Close'][0 : int(len(df) * 0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df) * 0.70) : int(len(df))])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

data_training_array = scaler.fit_transform(data_training)

#load model
model = load_model('keras_model.h5')

#testing part
past_100_days = data_training.tail(100)
final_df = past_100_days.append(data_testing, ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i-100, 0])
    
x_test, y_test = np.array(x_test), np.array(y_test)

y_predicted = model.predict(x_test)

scaler = scaler.scale_

scale_factor = 1 / scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

#final graph
st.subheader('Prediction vs Original')
fig2 = plt.figure(figsize=(12, 6))
plt.plot(y_test, 'b', label = 'Original Price')
plt.plot(y_predicted, 'r', label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)