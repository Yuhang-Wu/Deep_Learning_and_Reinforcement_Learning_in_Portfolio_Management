import pandas as pd


class config:

    # global
    raw_data = pd.read_csv("Stocks.csv", header=[0, 1], index_col=[0])
    tickers = ["AAPL", "GOOG", "JOBS", "LOGI", "MSFT"]  # select tickers

    # LSTM Model
    seq_len = 30  # history length
    prediction_len = 7  # prediction length
    train_proportion = 0.8  # ration of training data
    normalize = True

    lr = 0.001
    RNN_epochs = 20
    batch = 16
    validation_split = 0.1

    # CNN Price Prediction
    CNN_Epochs = 100

    # CNN Action Prediction
    history = 10
    n_stock = len(tickers)

    # Reinforcement Learning
    transaction_cost = 0.0002
    init_assets = 10000
    RL_episode = 50
