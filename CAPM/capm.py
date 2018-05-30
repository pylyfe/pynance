import pandas as pd 
import statsmodels.api as sm 

nke = pd.read_csv('NKE.csv', parse_dates = True, index_col = 'Date')
spy = pd.read_csv('^GSPC.csv', parse_dates = True, index_col = 'Date')

df = pd.concat([nke['Close'], spy['Close']], axis = 1)
df.columns = ['NKE', '^GSPC']

print(df.head())

M_Ret = df.pct_change(1)
M_Ret = M_Ret.dropna(axis = 0)
print(M_Ret.head())

x = M_Ret['^GSPC']
y = M_Ret['NKE']

x1 = sm.add_constant(x)

linreg = sm.OLS(y, x1)

results = linreg.fit()
print(results.summary())