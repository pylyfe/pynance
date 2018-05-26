import datetime as dt
import matplotlib.pyplot as plt
from  matplotlib import style 
import pandas as pd
import pandas_datareader.data as web

style.use('ggplot')

start = dt.datetime(2017, 1, 1)
end = dt.datetime.now()

# df = web.DataReader('KO', 'yahoo', start, end)

# Save data to CSV 

# df.to_csv('KO.csv')

# Read from CSV instead of from the Yahoo API

df = pd.read_csv('KO.csv', parse_dates = True, index_col = 0)

# Plot single variable 

# df['Adj Close'].plot()
# plt.show()

# # Plot multiple variables

# df[['High', 'Low']].plot()
# plt.show()

# Create a new column in the df to show Moving average 

df['100ma'] = df['Adj Close'].rolling(window = 100, min_periods = 0).mean()

# Graph Adj Close, 100 MA and Volume using subplots

# Create 2 subplots on a 6 x 1 matrix. The first subplot is 5 x 1
# and starts from the point (0,0), while the second subplot is 1 x 1
# and starts at the point (5, 0). Both also share the same x-axis

ax1 = plt.subplot2grid((6,1), (0,0), rowspan = 5, colspan = 1)
ax2 = plt.subplot2grid((6,1), (5,0), rowspan = 1, colspan = 1, sharex = ax1)

# Plot the graph 

# ax1.plot(df.index, df['Adj Close'])
# ax1.plot(df.index, df['100ma'])

# ax2.plot(df.index, df['Volume'])

# plt.show()

# Plot the data using a candlestick chart 

from matplotlib.finance import candlestick_ohlc
import matplotlib.dates as mdates

df_ohlc = df['Adj Close'].resample('10D').ohlc()
df_volume = df['Volume'].resample('10D').sum()

df_ohlc = df_ohlc.reset_index()

df_ohlc['Date'] = df_ohlc['Date'].map(mdates.date2num)

# Graph the candlestick chart

ax1 = plt.subplot2grid((6,1), (0,0), rowspan=5, colspan=1)
ax2 = plt.subplot2grid((6,1), (5,0), rowspan=1, colspan=1, sharex=ax1)
ax1.xaxis_date()

candlestick_ohlc(ax1, df_ohlc.values, width=5, colorup='g')
ax2.fill_between(df_volume.index.map(mdates.date2num), df_volume.values, 0)
plt.show()
