import bs4 as bs
import pickle
import requests 

# Create function to pull out S&P500 Tickers 

# BeautifulSoup turns source code into a BS Object that can be treated
# like a typical Python object

def save_sp500_tickers():
    # In case Wikipedia declines access to Python

    # headers = {'User-Agent': 'Mozilla/5.0 (X11; Linux i686) AppleWebKit/537.17 (KHTML, like Gecko) Chrome/24.0.1312.27 Safari/537.17'}
    # resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies',
    #                         headers=headers)
    resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table', {'class': 'wikitable sortable'})

    # Iterate through tickers and append to list 

    tickers = []
    for row in table.findAll('tr')[1:71]:
        ticker = row.findAll('td')[0].text
        tickers.append(ticker)

    # Dump tickers to pickle

    with open("sp500tickers.pickle", "wb") as f:
        pickle.dump(tickers, f)

    return tickers

save_sp500_tickers()

# Import additional libraries 

import os
import pandas_datareader.data as web
import datetime as dt 
import pandas as pd

def get_data_from_yahoo(reload_sp500=False):

#   # If requested, the function will re-pull S&P500 list, if not
#   # they will just load the pickle

    if reload_sp500:
        tickers = save_sp500_tickers()
    else:
        with open("sp500tickers.pickle", "rb") as f:
            tickers = pickle.load(f)
    if not os.path.exists('stock_dfs'):
        os.makedirs('stock_dfs')

    # Parse data from Yahoo just once and save it

    # Creating directory and store stock data per company

    start = dt.datetime(2010, 1, 1)
    end = dt.datetime.now()
    for ticker in tickers:
        # just in case your connection breaks, we'd like to save our progress!
        if not os.path.exists('stock_dfs/{}_data.csv'.format(ticker)):
            df = web.DataReader(ticker, 'morningstar', start, end)
            df.reset_index(inplace=True)
            df.set_index("date", inplace=True)
            df.to_csv('stock_dfs/{}.csv'.format(ticker))
        else:
            print('Already have {}'.format(ticker))

get_data_from_yahoo()

# Create a function to assess all the data together 

def compile_data():

    # Pull out previously made list of tickers and create an empty df

    with open("sp500tickers.pickle", "rb") as f:
        tickers = pickle.load(f)

    main_df = pd.DataFrame()

    # Iterate (limit to 70 companies)

    for count, ticker in enumerate(tickers[:70]):
        df = pd.read_csv('stock_dfs/{}_data.csv'.format(ticker))
        df.set_index('date', inplace = True)

        # Bonus: Adding columns into each stock df while iterating

        # df['{}_HL_pct_diff'.format(ticker)] = (df['high'] - df['low']) / df['low']
        # df['{}_daily_pct_chng'.format(ticker)] = (df['close'] - df['open']) / df['open']

        # In this project we are only interested in the Adj Close 

        df.rename(columns = {'close': ticker}, inplace = True)
        df.drop(['open', 'high', 'low', 'volume', 'Name'], 1, inplace = True)

        # Build the shared dataframe

        if main_df.empty:
            main_df = df
        else:
            main_df = main_df.join(df, how = 'outer')

        if count % 10 == 0:
            print(count)

    print(main_df.head())
    main_df.to_csv('sp500_joined_closes.csv')

compile_data()

# Visualizing Data using Matplotlib

import matplotlib.pyplot as plt
from matplotlib import style 
import numpy as np

style.use('ggplot')


# Create function to graph data

def visualize_data():
    df = pd.read_csv('sp500_joined_closes.csv')

    df_corr = df.corr()
    print(df_corr.head())
    df_corr.to_csv('sp500corr.csv')

    data1 = df_corr.values
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)

    # Create a heatmap color 

    heatmap1 = ax1.pcolor(data1, cmap=plt.cm.RdYlGn)
    fig1.colorbar(heatmap1)

    # Set ticker names so we know which company is which

    ax1.set_xticks(np.arange(data1.shape[1]) + 0.5, minor=False)
    ax1.set_yticks(np.arange(data1.shape[0]) + 0.5, minor=False)
    ax1.invert_yaxis()
    ax1.xaxis.tick_top()

    # Add company names to the currently nameless ticks

    column_labels = df_corr.columns
    row_labels = df_corr.index
    ax1.set_xticklabels(column_labels)
    ax1.set_yticklabels(row_labels)

    # Rotate graph 90 degrees

    plt.xticks(rotation=90)
    heatmap1.set_clim(-1,1)
    plt.tight_layout()
    #plt.savefig("correlations.png", dpi = (300))
    plt.show()

visualize_data()

# Preprocessing data for machine learning 

def process_data_for_labels(ticker):
    hm_days = 7
    df = pd.read_csv('sp500_joined_closes.csv', index_col=0)
    tickers = df.columns.values.tolist()
    df.fillna(0, inplace=True)

    # Grab the percent returns for the next seven days 

    for i in range(1, hm_days+1):
        df['{}_{}d'.format(ticker, i)] = (df[ticker].shift(-i) - df[ticker]) / df[ticker]
    df.fillna(0, inplace=True)
    return tickers, df

# Function that creates our labels -> Buy, Sell, Hold 
# If price > 2% in 7 days, buy, if < 2% sell. 
# If neither, then hold current position (whatever it is)


def buy_sell_hold(*args):
    cols = [c for c in args]
    requirement = 0.02
    for col in cols:
        if col > requirement:
            return 1
        if col < -requirement:
            return -1
    return 0

    # *args: future price change columns (any number of columns)
    # map this function to a pandas df (our "label")

# Making features and labels

from collections import Counter

# Take any ticker, create the needed dataset, create label column 

def extract_featuresets(ticker):
    tickers, df = process_data_for_labels(ticker)

    df['{}_target'.format(ticker)] = list(map( buy_sell_hold,
                                               df['{}_1d'.format(ticker)],
                                               df['{}_2d'.format(ticker)],
                                               df['{}_3d'.format(ticker)],
                                               df['{}_4d'.format(ticker)],
                                               df['{}_5d'.format(ticker)],
                                               df['{}_6d'.format(ticker)],
                                               df['{}_7d'.format(ticker)]))

    # Distribution 

    vals = df['{}_target'.format(ticker)].values.tolist()
    str_vals = [str(i) for i in vals]
    print('Data spread:', Counter(str_vals))

    # Clean up data 

    df.fillna(0, inplace=True)
    df = df.replace([np.inf, -np.inf], np.nan)
    df.dropna(inplace=True)

    # Convert stock prices to % changes

    df_vals = df[[ticker for ticker in tickers]].pct_change()
    df_vals = df_vals.replace([np.inf, -np.inf], 0)
    df_vals.fillna(0, inplace=True)

    # Features and labels

    X = df_vals.values
    y = df['{}_target'.format(ticker)].values
    return X, y, df

# Machine Learning

from sklearn import svm, cross_validation, neighbors
from sklearn.ensemble import VotingClassifier, RandomForestClassifier

def do_ml(ticker):
    X, y, df = extract_featuresets(ticker)

    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.25)

    clf = VotingClassifier([('lsvc', svm.LinearSVC()),
                            ('knn', neighbors.KNeighborsClassifier()),
                            ('rfor', RandomForestClassifier())])

    clf.fit(X_train, y_train)
    confidence = clf.score(X_test, y_test)
    print('accuracy:', confidence)
    predictions = clf.predict(X_test)
    print('predicted class counts:', Counter(predictions))
    print()
    print()
    return confidence

# Perform ML for every ticker inside the pickle

from statistics import mean

with open("sp500tickers.pickle","rb") as f:
    tickers = pickle.load(f)

accuracies = []
for count,ticker in enumerate(tickers):

    if count%10==0:
        print(count)

    accuracy = do_ml(ticker)
    accuracies.append(accuracy)
    print("{} accuracy: {}. Average accuracy:{}".format(ticker,accuracy,mean(accuracies)))