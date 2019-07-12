import pandas as pd
import matplotlib
import mpl_finance
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')


def stockPricePlot(ticker='300292'):
    # Strp 1. Load Data
    history = pd.read_csv("../Data/IntradayCN/" + ticker + '.csv',
                          parse_dates=True, index_col=0)

    # Step 2. Data Manipulation
    close = history['close']
    close = close.reset_index()
    close['timestamp'] = close['timestamp'].map(matplotlib.dates.date2num)

    ohlc = history[['open', 'high', 'low', 'close']].resample('1H').ohlc()
    ohlc = ohlc.reset_index()
    ohlc['timestamp'] = ohlc['timestamp'].map(matplotlib.dates.date2num)

    # Step 3. Plot Figures. Subplot 1: Scatter plot. Subplot 2: Candle Stick plot
    # 3.1 Subplot 1: Scatter Plot
    subplot1 = plt.subplot2grid((2, 1), (0, 0), rowspan=1, colspan=1)
    subplot1.xaxis_date()
    subplot1.plot(close['timestamp'], close['close'], 'b.')

    # 3.1 Subplot 1: Candle Stick Plot
    subplot2 = plt.subplot2grid(
        (2, 1), (1, 0), rowspan=1, colspan=1, sharex=subplot1)
    mpl_finance.candlestick_ohlc(ax=subplot2, quotes=ohlc.values,
                                 width=0.01, colorup='r', colordown='g')

    plt.show()


# stockPricePlot(ticker='300292')