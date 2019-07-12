from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras import optimizers
from numpy import newaxis

import matplotlib.pyplot as plt
from config import config
import pandas as pd
import numpy as np
import datetime
import itertools


def Normalizatoin(data_slice):
    """ Data Normalization """

    dataNormalized = []
    for slice in data_slice:
        normalized_slice = [(p / slice[0]) - 1 for p in slice]
        dataNormalized.append(normalized_slice)
    return dataNormalized


def Denormalization(data_norm, data):
    """ Data Denormalization """

    dataDenormalized = []
    wholelen = 0
    for i, rowdata in enumerate(data_norm):
        denormalize = []

        if isinstance(rowdata, float) | isinstance(rowdata, np.float32):
            denormalize = [(rowdata + 1) * float(data[wholelen][0])]
            dataDenormalized.append(denormalize)
            wholelen += 1
        else:
            for j in range(len(rowdata)):
                denormalize.append((rowdata[j] + 1) * data[wholelen][0])
                wholelen += 1
            dataDenormalized.append(denormalize)
    return dataDenormalized


def load_data(ticker, normalize=True):
    """ load data from a csv. file, return training and testing dataset """

    # Step 1. select stock daily Adj. Closed price
    daily_price = config.raw_data[ticker]

    dates = list(daily_price.index)
    closeData = list(daily_price["Adj Close"])
    seq_len = config.seq_len + 1

    # Step 2. data normalize
    data_segment, date_segment = [], []
    for i in range(len(closeData) - seq_len):
        data_segment.append(closeData[i: i + seq_len])
        date_segment.append(dates[i: i + seq_len])
    norm_data = Normalizatoin(data_segment) if normalize else data_segment

    data_segment = np.array(data_segment)
    date_segment = np.array(date_segment)
    norm_data = np.array(norm_data)

    # Step 3. separate the data into training and testing set
    row = round(config.train_proportion * norm_data.shape[0])
    dates_train = date_segment[: row, :]
    train = norm_data[: row, :]
    np.random.shuffle(train)

    x_train, y_train = train[:, : -1], train[:, -1]
    x_tr_date, y_tr_date = dates_train[:, : -1], dates_train[:, -1]

    x_test, y_test = norm_data[row:, :-1], norm_data[row:, -1]
    x_te_date, y_te_date = date_segment[row:, :-1], date_segment[row:, -1]

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # Step 4. define test data
    test_data = data_segment[row:, :]

    return x_train, y_train, x_test, y_test, y_te_date, test_data


def RNN_LSTM():
    """ Build a Long Short-Term Model """

    # initialize model
    model = Sequential()

    # 1st LSTM & Dropout layer
    model.add(LSTM(input_shape=(None, 1),
                   units=config.seq_len,
                   return_sequences=True))
    model.add(Dropout(0.2))

    # 2nd LSTM & Dropout layer
    model.add(LSTM(units=100, return_sequences=False))
    model.add(Dropout(0.2))

    # Output layer
    model.add(Dense(units=1))
    model.add(Activation("linear"))

    # optimizer
    opt = optimizers.RMSprop(lr=config.lr, rho=0.9, epsilon=1e-06)

    # compile model
    model.compile(loss="mean_squared_error", optimizer=opt)

    return model


def predict_point_by_point(model, data):
    """ predict one step at a time """

    predicted = model.predict(data)
    predicted = np.reshape(predicted, (predicted.size))
    return predicted


def predict_sequence_full(model, data):
    """ Gradually predict the entire time series based on the training model and
        the length of the time series used for prediction in the first segment """

    curr_frame = data[0]
    predicted = []

    for i in range(len(data)):
        predicted.append(model.predict(curr_frame[newaxis, :, :])[0, 0])
        curr_frame = curr_frame[1:]
        curr_frame = np.insert(
            curr_frame, [config.seq_len - 1], predicted[-1], axis=0)

    return predicted


def predict_sequences_multiple(model, data):
    """ A sequence that predicts the length of the prediction_len step by step based on the training model
        and the length of the time series used to predict each segment """

    prediction_seqs = []
    for i in range(int(len(data) / config.prediction_len)):
        curr_frame = data[i * config.prediction_len]

        predicted = []
        for j in range(config.prediction_len):
            predicted.append(model.predict(curr_frame[newaxis, :, :])[0, 0])
            curr_frame = curr_frame[1:]
            curr_frame = np.insert(curr_frame, [config.seq_len - 1],
                                   predicted[-1], axis=0)
        prediction_seqs.append(predicted)

    return prediction_seqs


def plot_result(pt, seq, multi, true_data, ticker):
    """ plot the predicted and true data """

    fig = plt.figure(facecolor='white')
    fig, ax = plt.subplots(2, figsize=(18, 8))

    ax[0].plot(true_data, label='True Data')
    ax[0].plot(pt, label='Predict one step at a time')
    ax[0].plot(seq, label='Predict entire time series')
    ax[0].legend()

    ax[1].plot(true_data, label='True Data')
    for i, data in enumerate(multi):
        padding = [None for p in range(i * config.prediction_len)]
        ax[1].plot(padding + data)
    ax[1].legend()

    plt.suptitle(
        "Compare the Predicted Stock Price (%s) with the True Price" % ticker, fontsize=18)
    plt.show()


def LSTM_main(ticker):
    """ main function of the LSTM Model """

    # Step 1. build training and testing data
    X_train, y_train, X_test, y_test, date_test, data_test = load_data(ticker)

    # Step 2. build model
    model = RNN_LSTM()

    # Step 3. train model
    model.fit(X_train, y_train,
              batch_size=config.batch,
              epochs=config.RNN_epochs,
              validation_split=config.validation_split)

    # Step 4. stock price prediction
    y_pred_pt = predict_point_by_point(model, X_test)
    y_pred_seq = predict_sequence_full(model, X_test)
    y_pred_multi = predict_sequences_multiple(model, X_test)

    # Step 5. denormalize data
    if config.normalize:
        y_hat_pt = Denormalization(y_pred_pt, data_test)
        y_hat_seq = Denormalization(y_pred_seq, data_test)
        y_hat_multi = Denormalization(y_pred_multi, data_test)
        y_test = Denormalization(y_test, data_test)

    return y_hat_pt, y_hat_seq, y_hat_multi, y_test, date_test


def mixed_Predicted_Price():
    """ rearrange all the selected stocks predicted price """

    y_true = [[] for _ in config.tickers]
    pt_, seq_, multi_ = y_true.copy(), y_true.copy(), y_true.copy()
    pt_df, seq_df, multi_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    for i, ticker in enumerate(config.tickers):
        pt_[i], seq_[i], multi_[i], y_true[i], date_test = LSTM_main(ticker)

        # unstack the list
        pt_df[ticker] = list(itertools.chain(*pt_[i]))
        seq_df[ticker] = list(itertools.chain(*seq_[i]))
        multi_df[ticker] = list(itertools.chain(*multi_[i]))

    pt_df.index = date_test
    seq_df.index = date_test
    multi_df.index = date_test[:-4]

    return pt_, seq_, multi_, pt_df, seq_df, multi_df, y_true


# # Run LSTM Model: Stock Price Prediction
# pt_, seq_, multi_, pt_df, seq_df, multi_df, y_true = mixed_Predicted_Price()
#
# # Plot the Predicted Stock Price & True Price
# plot_result(pt_[0], seq_[0], multi_[0], y_true[0], config.tickers[0])
# plot_result(pt_[1], seq_[1], multi_[1], y_true[1], config.tickers[1])
# plot_result(pt_[2], seq_[2], multi_[2], y_true[2], config.tickers[2])
# plot_result(pt_[3], seq_[3], multi_[3], y_true[3], config.tickers[3])
# plot_result(pt_[4], seq_[4], multi_[4], y_true[4], config.tickers[4])
