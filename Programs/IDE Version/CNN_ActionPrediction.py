import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import LSTM_Model
from config import config
from draw_step import drawHist
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, AveragePooling2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.regularizers import l2
from sklearn.preprocessing import StandardScaler


def load_data(dataset, train):
    """ load stock data """

    if train:
        daily_close = pd.DataFrame(index=dataset.index)

        for ticker in config.tickers:
            daily_close[ticker] = dataset[ticker]["Adj Close"]
    else:
        daily_close = dataset
    daily_lrt = np.log(1 + daily_close.pct_change().dropna())

    # separate the date by history legth
    train_num = daily_lrt.shape[0] - config.history + 1
    X = np.empty((train_num, config.history, config.n_stock))
    Y = np.array(daily_lrt[:train_num])
    Y[Y < 0] = 0
    for i in range(train_num):
        X[i] = daily_lrt.iloc[i:config.history + i].values
        # portfolio percentage
        Y[i] = np.zeros(5) if (Y[i] <= 0).all() else np.true_divide(
            Y[i], Y[i].sum(keepdims=True))

    daily_close_train = daily_close[:train_num]

    return X, Y, daily_close_train


def CNN_Action():
    """ build CNN Model: predict the stock proportion in th portfolio"""

    model = Sequential()

    # 1st CNN & Dropout layer
    model.add(Conv2D(filters=10, kernel_size=2,
                     input_shape=(config.history, config.n_stock, 1),
                     strides=(1, 1), padding="same", activation="relu"))
    model.add(AveragePooling2D(pool_size=(1, 1), strides=1))
    model.add(Dropout(0.2))

    # 2nd CNN & Dropout layer
    model.add(Conv2D(filters=64, kernel_size=2,
                     strides=(1, 1), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 1), strides=2))
    model.add(Dropout(0.2))

    # 3rd CNN & Dropout layer
    model.add(Conv2D(filters=config.n_stock, kernel_size=2,
                     strides=(1, 1), padding="same", activation="relu"))
    model.add(Dropout(0.2))

    # output layer
    model.add(Flatten())
    model.add(Dense(units=config.n_stock,
                    kernel_initializer="he_normal",
                    kernel_regularizer=l2(0.001)))
    model.add(Activation("softmax"))

    # compile model
    model.compile(optimizer="adam", loss="mean_squared_error",
                  metrics=["accuracy"])

    return model


def actionPrediction():

    # Step 1. load training data
    dataset = config.raw_data[config.tickers]
    X, Y, daily_close = load_data(dataset, True)
    x_test, y_test, _ = load_data(pt_df, False)

    # define the training & testing dates
    dates = np.array(dataset.index)
    _, _, _, _, date_test, _ = LSTM_Model.load_data(config.tickers[0])
    date_train = np.setdiff1d(dates, date_test, assume_unique=True)

    daily_close_train = daily_close[:date_train.shape[0]]
    daily_close_test = daily_close[date_train.shape[0]:]

    x_train, y_train = X[:date_train.shape[0]], Y[:date_train.shape[0]]

    # Step 3. build model
    model = CNN_Action()

    # Step 4. fit model
    x_train = np.reshape(x_train, [x_train.shape[0], x_train.shape[1],
                                   x_train.shape[2], 1])
    hist = model.fit(x_train, y_train, batch_size=8,
                     epochs=config.CNN_Epochs, verbose=False)

    # Step 5. predict
    x_test = np.reshape(x_test, [x_test.shape[0], x_test.shape[1],
                                 x_test.shape[2], 1])
    y_pred = model.predict(x_test)

    # Step 6. Plot the learning step
    drawHist(pd.DataFrame(hist.history))

    return y_pred, daily_close_test


# # Run CNN Model: Compute Stock Proportion
# y_pred, daily_close_test = actionPrediction()
