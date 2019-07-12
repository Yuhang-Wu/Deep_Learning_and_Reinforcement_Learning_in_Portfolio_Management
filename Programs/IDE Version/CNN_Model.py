import talib
import numpy as np
import pandas as pd
from config import config
from draw_step import drawHist

from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Dropout, Activation, Dropout
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt


def addFeature(dataset):
    """ add stock feature base on the dataset """

    dataset["H_L"] = dataset["High"] - dataset["Low"]
    dataset['O_C'] = dataset['Adj Close'] - dataset['Open']
    dataset["K_L"] = (dataset["Adj Close"] - dataset["Open"]) / dataset["H_L"]
    dataset["OBV"] = dataset["Volume"] * \
        (dataset["Adj Close"] * 2 - dataset["H_L"]) / dataset["H_L"]

    dataset['3day MA'] = dataset['Adj Close'].shift(1).rolling(window=3).mean()
    dataset['10day MA'] = dataset['Adj Close'].shift(
        1).rolling(window=10).mean()
    dataset['30day MA'] = dataset['Adj Close'].shift(
        1).rolling(window=30).mean()
    dataset['Std_dev'] = dataset['Adj Close'].rolling(5).std()

    dataset['RSI'] = talib.RSI(dataset['Adj Close'].values, timeperiod=9)
    dataset['Williams %R'] = talib.WILLR(
        dataset['High'].values, dataset['Low'].values, dataset['Adj Close'].values, 7)
    dataset['Price_Rise'] = np.where(
        dataset['Adj Close'].shift(-1) > dataset['Adj Close'], 1, 0)

    dataset = dataset.dropna()

    return dataset


def buildModel(n_feature=10):
    """ build neural network model """

    # Initialization
    model = Sequential()

    # 1st Dense & Dropout Layer
    model.add(Dense(units=64,
                    kernel_initializer='uniform',
                    activation="relu",
                    input_dim=n_feature))
    model.add(Dropout(0.2))

    # 2nd Dense & Dropout Layer
    model.add(Dense(units=128,
                    kernel_initializer='uniform',
                    activation="relu"))
    model.add(Dropout(0.2))

    # Output layer
    model.add(Dense(units=1,
                    kernel_initializer='uniform',
                    activation='sigmoid'))

    # Compile layer
    model.compile(optimizer='adam',
                  loss='mean_squared_error',
                  metrics=['accuracy'])

    return model


def stockPriceTrend():
    """ main function"""

    # Step 1. read data from csv and add features
    stock_features = addFeature(config.raw_data[config.tickers[0]])

    X = stock_features.iloc[:, 5:-1]  # features dataframe
    y = stock_features.iloc[:, -1]  # price change

    # Step 2. define training / testing datset
    split = int(len(stock_features) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Step 3. Data Normalization
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # Step 4. Build Model
    model = buildModel()
    trend_hist = model.fit(
        X_train, y_train, batch_size=32, epochs=config.CNN_Epochs)

    # Step 5. Predict Stock Trend

    y_pred = model.predict(X_test)
    y_pred = (y_pred > 0.5)

    stock_features['y_pred'] = np.NaN
    stock_features.iloc[(len(stock_features) - len(y_pred)):, -1:] = y_pred
    stock_features = stock_features.dropna()

    # Step 6. Cummulative Trturn
    stock_features['Tomorrows Returns'] = 0.
    stock_features['Tomorrows Returns'] = np.log(
        stock_features['Adj Close'] / stock_features['Adj Close'].shift(1))
    stock_features['Tomorrows Returns'] = stock_features['Tomorrows Returns'].shift(
        -1)

    stock_features['Strategy Returns'] = 0.
    stock_features['Strategy Returns'] = np.where(stock_features['y_pred'] == True,
                                                  stock_features['Tomorrows Returns'],
                                                  - stock_features['Tomorrows Returns'])

    stock_features['Cumulative Market Returns'] = np.cumsum(
        stock_features['Tomorrows Returns'])
    stock_features['Cumulative Strategy Returns'] = np.cumsum(
        stock_features['Strategy Returns'])

    return stock_features, trend_hist


# # Run CNN Model: Predict cummulative return
# cummulative_rt, trend_hist = stockPriceTrend()
#
# # Step 7. Plot learning steps
# drawHist(pd.DataFrame(trend_hist.history))
