# %% Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# %% Import data
dtypes = [str, float]

dataframe = pd.read_csv('data/airline-passengers.csv', usecols=[1], engine='python',
                        dtype={'Passengers': 'float'})

dataset = dataframe.values

dataframe.plot()
# %%
training_data_len = int(len(dataset) * 0.7)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

train_data = scaled_data[0:training_data_len, :]

testing_data_len = len(dataset) - training_data_len
test_data = scaled_data[training_data_len:, :]

# %% prepare batchees of data for time series analysis


def create_batch_dataset(dataframe, look_back=1):
    BATCH_SIZE = look_back
    x_train = []
    y_train = []
    NEXT_PREDICTION_STEP = 1

    for i in range(BATCH_SIZE, len(dataframe) - NEXT_PREDICTION_STEP):
        x_train.append(dataframe[i - BATCH_SIZE: i, :])
        y_train.append(dataframe[i + NEXT_PREDICTION_STEP, 0])

    return np.array(x_train), np.array(y_train)


look_back = 6
x_train, y_train = create_batch_dataset(train_data, look_back=6)

# %% Build the LSTM network model
model = Sequential()
batch_size = 1
model.add(LSTM(4, batch_input_shape=(batch_size, look_back, 1),
          stateful=True, return_sequences=True))
model.add(LSTM(4, batch_input_shape=(batch_size, look_back, 1), stateful=True))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

# %% Train the model
# stoppingCallback = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=10)

history = model.fit(
    x_train,
    y_train,
    batch_size=1,
    epochs=30,
    # callbacks=[stoppingCallback],
    shuffle=False,
)

# %% prepare test data

x_test, y_test = create_batch_dataset(test_data, look_back=6)

# %% Run prediction

x_train_predict = model.predict(x_train, batch_size=batch_size)
model.reset_states()
x_test_predict = model.predict(x_test, batch_size=batch_size)

x_train_predict = scaler.inverse_transform(x_train_predict)
y_train = scaler.inverse_transform([y_train])
x_test_predict = scaler.inverse_transform(x_test_predict)
y_test = scaler.inverse_transform([y_test])


# calculate root mean squared error
trainScore = np.math.sqrt(mean_squared_error(
    y_train[0], x_train_predict[:, 0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = np.math.sqrt(mean_squared_error(y_test[0], x_test_predict[:, 0]))
print('Test Score: %.2f RMSE' % (testScore))


# %% Plot results
# shift train predictions for plotting
trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(x_train_predict)+look_back, :] = x_train_predict

# shift test predictions for plotting
testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
testPredictPlot[-len(x_test_predict):, :] = x_test_predict

# %% plot baseline and predictions
# plt.plot(scaler.inverse_transform(dataset))
plt.plot(dataset)
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()
