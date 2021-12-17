# Program uses LSTM to predict the closing stock price of Apple Inc. using 60 past 60 day stock price

#Importing libraries
import math
import numpy as np
import pandas as pd
import pandas_datareader as web
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

#Get the stock quote
df = web.DataReader('AAPL', data_source='yahoo', start='2016-01-01', end='2021-12-1')

# plt.figure(figsize=(16,8))
# plt.plot(df['Close'])
# plt.xlabel('Date', fontsize=18)
# plt.ylabel('Close Price USD ($)', fontsize=18)
# plt.show()

#Create dataframe with only the 'Close' column
data = df.filter(['Close'])

#Convert dataframe to numpy array
dataset = data.values

#Getting the number of rows to train the model on
training_data_len = math.ceil(len(dataset)*.8)

#Scale the data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

#Create scaled training dataset
train_data = scaled_data[0:training_data_len, :]

#Splitting into X_train and y_train
X_train = []
y_train = []

for i in range(60, len(train_data)):
    X_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])

#Convert X_train and y_train to numpy arrays
X_train, y_train = np.array(X_train), np.array(y_train)

#Reshape
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

#Build LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

#Compile model
model.compile(optimizer='adam', loss='mean_squared_error')

#Train model
model.fit(X_train, y_train, batch_size=1, epochs=1)

#Create scaled testing dataset
test_data = scaled_data[training_data_len - 60: , :]

#Create X_test and y_test
X_test = []
y_test = dataset[training_data_len:, :]

for i in range(60, len(test_data)):
    X_test.append(test_data[i-60:i, 0])

#Convert test dataset to numpy array and reshape
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

#Get the model's predicted price values
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)

#Get RMSE for evaluation
rmse=np.sqrt(np.mean(((predictions-y_test)**2)))

#Plot the data, creating validation dataset
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions

#Visualise data
def plot_data():
    plt.figure(figsize=(16,8))
    plt.title('Model')
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price USD ($)', fontsize=18)
    plt.plot(train['Close'])
    plt.plot(valid[['Close', 'Predictions']])
    plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
    plt.show()

plot_data()

#Get the quote
apple_quote = web.DataReader('AAPL', data_source='yahoo', start='2016-01-01', end='2021-12-1')

#Create new dataframe
new_df = apple_quote.filter(['Close'])

#Get last 60 day closing price values and convert df to array
last_60_days = new_df[-60:].values

#Scale between 0 and 1
last_60_days_scaled = scaler.transform(last_60_days)

#Creating testing dataset and converting it to np array and reshaping it
X_test_60 = []
X_test_60.append(last_60_days_scaled)
X_test_60 = np.array(X_test_60)
X_test_60 = np.reshape(X_test_60, (X_test_60.shape[0], X_test_60.shape[1], 1))

#Get predicted scaled price
pred_price = model.predict(X_test_60)

#Undo scaling
pred_price = scaler.inverse_transform(pred_price)

#Get quote
apple_quote2 = web.DataReader('AAPL', data_source='yahoo', start='2021-12-01', end='2021-12-1')
#Print predicted price vs actualy prive
print(pred_price, apple_quote2['Close'])