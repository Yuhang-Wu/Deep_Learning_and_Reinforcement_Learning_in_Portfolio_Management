import os
import datetime

import pandas as pd
from yahoo_historical import Fetcher

DATA = ["Open", "High", "Low", "Adj Close", "Volume"]


def DataDownloader():

    start, end = [2010, 1, 1], [2019, 4, 1]

    # keep the first row if duplicate company name
    companies = pd.read_csv("companylist.csv").drop_duplicates(
        subset='Name', keep="first")
    companies = companies[companies.Sector == "Technology"]

    Symbols = [symbol.rstrip() for symbol in companies.Symbol]

    Stocks = Fetcher(Symbols[0], start, end).getHistorical(
    ).set_index("Date").rename_axis(None)[DATA]

    Stocks.index = pd.DatetimeIndex(Stocks.index)
    Stocks.columns = pd.MultiIndex.from_product([[Symbols[0]], Stocks.columns])

    for s in Symbols[1:]:

        df = Fetcher(s, start, end).getHistorical(
        ).set_index("Date").rename_axis(None)[DATA]

        df.index = pd.DatetimeIndex(df.index)
        df.columns = pd.MultiIndex.from_product([[s], df.columns])

        if df.index[0] <= datetime.datetime(2010, 1, 4):
            Stocks = pd.merge(Stocks, df, on=Stocks.index).set_index(
                "key_0").rename_axis(None)

    Stocks.to_csv(os.getcwd() + "/Stocks.csv")


# DataDownloader()
