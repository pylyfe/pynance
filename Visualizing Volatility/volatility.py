import pandas as pd
from pandas_datareader import data as web
import matplotlib.pyplot as plt
from datetime import datetime

start = datetime(2017,1,1)
end = datetime(2018,1,1)

# FANG stocks
assets = ['AAPL', 'FB', 'NFLX', 'GOOGL']

df = pd.DataFrame()

for stock in assets:
    df[stock] = web.DataReader(stock, data_source='quandl', start=start, end=end)['AdjClose']

asset_returns_daily = df.pct_change()
asset_volatility_daily = asset_returns_daily.std()

asset_returns_daily.plot(figsize = (10,6)) # .hist(bins = 50, figsize = (10,6))
plt.xlabel('Daily Returns')
plt.show()

asset_returns_daily.plot.hist(bins = 50, figsize = (10,6))
plt.xlabel('Daily Returns Histogram')
plt.show()