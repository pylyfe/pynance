# import python's number cruncher
import pandas_datareader.data as web
import pandas as pd
import numpy as np

assets =  ['KO', 'PEP', 'GE', 'ORCL', 'WMT'] 

df = pd.DataFrame()  

for stock in assets:
    df[stock] = web.DataReader(stock, data_source='yahoo',
                               start='2015-1-1' , end='2017-1-1')['Adj Close']

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

percent_var = str(round(port_variance, 4) * 100) + '%'
percent_vols = str(round(port_volatility, 4) * 100) + '%'

print('Variance of Portfolio is {}, Portfolio Risk is {}'
      .format(percent_var, percent_vols))

