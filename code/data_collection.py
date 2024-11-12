# This file gets data for stock pairs and generates synthetic data for training


import numpy as np
import pandas as pd
import yfinance as yf

def download_stock_data(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    data = data.dropna().reset_index(drop=True)
    return data

def generate_gbm(start_price, mu, sigma, days):
    dt = 1/252
    prices = []
    price = start_price
    for _ in range(days):
        price *= np.exp((mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * np.random.normal())
        prices.append(price)
    return prices

def generate_synthetic_data(length):
    np.random.seed(42)
    synthetic_data = pd.DataFrame({
        'Asset1': generate_gbm(100, 0.05, 0.2, length),
        'Asset2': generate_gbm(100, 0.03, 0.25, length)
    })
    return synthetic_data.reset_index(drop=True)

if __name__ == "__main__":
    tickers = ['GS', 'JPM']
    start_date = '2008-01-01'
    end_date = '2011-01-01'
    real_data = download_stock_data(tickers, start_date, end_date)
    real_data.to_csv('data/real_data.csv', index=False)
    synthetic_length = len(real_data)
    synthetic_data = generate_synthetic_data(synthetic_length)
    synthetic_data.to_csv('data/synthetic_data.csv', index=False)
    data = pd.concat([real_data.reset_index(drop=True), synthetic_data], axis=1)
    data.to_csv('data/combined_data.csv', index=False)
