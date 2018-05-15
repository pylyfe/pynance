import pandas as pd
import matplotlib.pyplot as plt

def create_csv(name):
    file_path = f'{name}_price.csv'
    return pd.read_csv(file_path, parse_dates=True, index_col='Date')


dataset_to_create = ['bitcoin', 'bitcoin_cash', 'dash', 'ethereum_classic',
                     'bitconnect', 'litecoin', 'monero', 'nem',
                     'neo', 'numeraire', 'omisego',
                     'qtum', 'ripple', 'stratis', 'waves']

cryptos = [create_csv(currency) for currency in dataset_to_create]

bitcoin = cryptos[0]
bitcoin_cash = cryptos[1]
dash = cryptos[2]
ethereum_classic = cryptos[3]
bitconnect = cryptos[4]
litecoin = cryptos[5]
monero = cryptos[6]
nem = cryptos[7]
neo = cryptos[8]
numeraire = cryptos[9]
omisego = cryptos[10]
qtum = cryptos[11]
ripple = cryptos[12]
stratis = cryptos[13]
waves = cryptos[14]

dataset = [bitcoin, bitcoin_cash, dash,
           ethereum_classic, bitconnect,
           litecoin, monero, nem, neo,
           numeraire, omisego, qtum,
           ripple,stratis, waves]

for item in dataset:
    item.sort_index(inplace=True)
    item['30_day_mean'] = item['Close'].rolling(window=30).mean()
    item['50_day_mean'] = item['Close'].rolling(window=50).mean()
    item['100_day_mean'] = item['Close'].rolling(window=100).mean()
    item['30_day_volatility'] = item['Close'].rolling(window=30).std()

# Graph Every Cryptocurrency (this is for the purpose of printing)

bitcoin[['Close','30_day_mean', '50_day_mean', '100_day_mean', '30_day_volatility']].plot(figsize=(10,8));
plt.title('Bitcoin Closing Price with 30 Day Mean & Volatility')
plt.ylabel('Price')
plt.show()

bitcoin_cash[['Close','30_day_mean', '50_day_mean', '100_day_mean', '30_day_volatility']].plot(figsize=(10,8));
plt.title('Bitcoin Cash Closing Price with 30 Day Mean & Volatility')
plt.ylabel('Price')
plt.show()

dash[['Close','30_day_mean', '50_day_mean', '100_day_mean', '30_day_volatility']].plot(figsize=(10,8));
plt.title('Dash Closing Price with 30 Day Mean & Volatility')
plt.ylabel('Price')
plt.show()

ethereum_classic[['Close','30_day_mean', '50_day_mean', '100_day_mean','30_day_volatility']].plot(figsize=(10,8));
plt.title('Ethereum Closing Price with 30 Day Mean & Volatility')
plt.ylabel('Price')
plt.show()

bitconnect[['Close','30_day_mean', '50_day_mean', '100_day_mean','30_day_volatility']].plot(figsize=(10,8));
plt.title('Bitconnect Closing Price with 30 Day Mean & Volatility')
plt.ylabel('Price')
plt.show()

litecoin[['Close','30_day_mean', '50_day_mean', '100_day_mean','30_day_volatility']].plot(figsize=(10,8));
plt.title('Litecoin Closing Price with 30 Day Mean & Volatility')
plt.ylabel('Price')
plt.show()

monero[['Close','30_day_mean', '50_day_mean', '100_day_mean','30_day_volatility']].plot(figsize=(10,8));
plt.title('Monero Closing Price with 30 Day Mean & Volatility')
plt.ylabel('Price')
plt.show()

nem[['Close','30_day_mean', '50_day_mean', '100_day_mean','30_day_volatility']].plot(figsize=(10,8));
plt.title('NEM Closing Price with 30 Day Mean & Volatility')
plt.ylabel('Price')
plt.show()

neo[['Close','30_day_mean', '50_day_mean', '100_day_mean','30_day_volatility']].plot(figsize=(10,8));
plt.title('NEO Closing Price with 30 Day Mean & Volatility')
plt.ylabel('Price')
plt.show()

numeraire[['Close','30_day_mean', '50_day_mean', '100_day_mean','30_day_volatility']].plot(figsize=(10,8));
plt.title('Numeraire Closing Price with 30 Day Mean & Volatility')
plt.ylabel('Price')
plt.show()

omisego[['Close','30_day_mean', '50_day_mean', '100_day_mean','30_day_volatility']].plot(figsize=(10,8));
plt.title('Omisego Closing Price with 30 Day Mean & Volatility')
plt.ylabel('Price')
plt.show()

qtum[['Close','30_day_mean', '50_day_mean', '100_day_mean','30_day_volatility']].plot(figsize=(10,8));
plt.title('Qtum Closing Price with 30 Day Mean & Volatility')
plt.ylabel('Price')
plt.show()

ripple[['Close','30_day_mean', '50_day_mean', '100_day_mean','30_day_volatility']].plot(figsize=(10,8));
plt.title('Ripple Closing Price with 30 Day Mean & Volatility')
plt.ylabel('Price')
plt.show()

stratis[['Close','30_day_mean', '50_day_mean', '100_day_mean','30_day_volatility']].plot(figsize=(10,8));
plt.title('Stratis Closing Price with 30 Day Mean & Volatility')
plt.ylabel('Price')
plt.show()

waves[['Close','30_day_mean', '50_day_mean', '100_day_mean','30_day_volatility']].plot(figsize=(10,8));
plt.title('Waves Closing Price with 30 Day Mean & Volatility')
plt.ylabel('Price')
plt.show()

# More concise way of graphing

# for item in dataset:
#   item[['Close','30_day_mean', '50_day_mean', '100_day_mean','30_day_volatility']].plot(figsize=(10,8));
#   plt.title('Closing Price with 30 Day Mean & Volatility')
#   plt.ylabel('Price')
#   plt.show()

# Frequency: Still have no idea what this is 

# print(bitcoin['Close'].resample('A').mean())
# print(bitcoin['Close'].resample('A').apply(lambda x: x[-1]))