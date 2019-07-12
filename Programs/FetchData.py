import io
import os
import requests
import datetime
import tushare
import pandas as pd

dateToday = datetime.datetime.today().strftime("%Y%m%d")

#
# def dataframeFromUrl(url):
#     dataString = requests.get(url).content
#     parseResult = pd.read_csv(io.StringIO(
#         dataString.decode('utf-8')), index_col=0)
#     return parseResult


# def stockPriseIntraday_US(ticker, folder):
#     # Step 1. Get intraday data online
#     url_ = "https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={ticker}&interval=1min&outputsize=full&apikey=6KIYZMX2GDJA1OWG&datatype=csv".format(
#         ticker=ticker)
#     intraday = dataframeFromUrl(url_)
#
#     # Step 2. append if history exists
#     file = folder + '/' + ticker + '.csv'
#
#     if os.path.exists(file):
#         history = pd.read_csv(file, index_col=0)
#         intraday.append(history)
#
#     # Step 3. Inverse on index
#     intraday.sort_index(inplace=True)
#
#     # step 4: Save
#     intraday.to_csv(file)


def stockPriseIntraday_CN(ticker, folder):
    # Step 1. Get intraday data online
    intraday = tushare.get_hist_data(ticker, ktype='5')

    # Step 2. append if history exists
    file_ = folder + '/' + ticker + '.csv'

    if os.path.exists(file_):
        history = pd.read_csv(file_, index_col=0)
        intraday.append(history)

    # Step 3. Inverse on index
    intraday.sort_index(inplace=True)
    intraday.index.name = 'timestamp'

    # step 4: Save
    intraday.to_csv(file_)


def tickerRawData():
    # # Step 1. Get tickers list
    # # US
    # url_US = "https://www.nasdaq.com/screening/companies-by-industry.aspx?industry=Technology&exchange=NASDAQ&render=download"
    # dataString = requests.get(url_US).content
    # tickerRawData_US = pd.read_csv(io.StringIO(dataString.decode('utf-8')))
    # tickers_US = tickerRawData_US["Symbol"].tolist()

    # CN
    tickerRawData_CN = tushare.get_stock_basics()
    tickers_CN = tickerRawData_CN.index.tolist()

    # # Step 2. Save the tickers list to a local file
    # # US
    # file_US = "../Data/TickerListUS" + dateToday + '.csv'
    # tickerRawData_US.to_csv(file_US, index=False)

    # CN
    file_CN = "../Data/TickerListCN" + dateToday + '.csv'
    tickerRawData_CN.to_csv(file_CN)


# Step 3. Get stock price (intraday)
# # US
# for i, ticker in enumerate(tickers_US):
#     try:
#         print("IntradayUS", i, )
#         stockPriseIntraday_US(ticker, folder="../Data/IntradayUS")
#     except:
#         pass

# CN
for i, ticker in enumerate(tickers_CN):
    try:
        stockPriseIntraday_CN(ticker, folder="../Data/IntradayCN")
    except:
        pass
