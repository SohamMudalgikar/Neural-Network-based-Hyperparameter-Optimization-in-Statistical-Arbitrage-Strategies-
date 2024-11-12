# Compute additional statistical indicators like spread, volatility, and correlation.

import pandas as pd

def feature_engineering(data, tickers):
    data['Spread'] = data[tickers[0]] - data[tickers[1]]
    window_size = min(20, len(data) - 1)
    data['Spread_vol'] = data['Spread'].rolling(window=window_size).std()
    data['Correlation'] = data[tickers[0]].rolling(window=window_size).corr(data[tickers[1]])
    data['Spread_vol'].fillna(method='bfill', inplace=True)
    data['Correlation'].fillna(method='bfill', inplace=True)
    data.dropna(inplace=True)
    data.reset_index(drop=True, inplace=True)
    return data

if __name__ == "__main__":
    tickers = ['GS', 'JPM']
    data = pd.read_csv('data/combined_data.csv')
    data = feature_engineering(data, tickers)
    data.to_csv('data/processed_data.csv', index=False)
