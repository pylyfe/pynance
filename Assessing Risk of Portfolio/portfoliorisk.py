# import python's number cruncher
import pandas_datareader.data as web
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

assets =  ['KO', 'PEP', 'GE', 'ORCL', 'WMT'] 

df = pd.DataFrame()  

for stock in assets:
    df[stock] = web.DataReader(stock, data_source='quandl',
                               start='2015-1-1' , end='2017-1-1')['AdjClose']

# Get Daily Returns

d_returns = df.pct_change()  

print(df.head())

cov_matrix_d = d_returns.cov() #Daily covariance
cov_matrix_a = cov_matrix_d * 250 #Annual covariance

weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])  # assign equal weights

# calculate the variance and risk of the portfolo
port_variance = np.dot(weights.T, np.dot(cov_matrix_a, weights))

# Standard Deviation (Volatility)
port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix_a, weights)))

percent_var = port_variance * 100
percent_vols = port_volatility * 100

print('Variance of Portfolio is %.2f, Portfolio Risk is %.2f' % (percent_var, percent_vols))

d_returns.plot(figsize=(16,8))

plt.show()

print(d_returns.describe())

print(d_returns.corr())
