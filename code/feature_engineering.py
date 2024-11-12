# Compute additional statistical indicators like spread, volatility, and correlation.

# Calculate the spread between the two assets
data['Spread'] = data[tickers[0]] - data[tickers[1]]

# Rolling volatility of the spread
data['Spread_vol'] = data['Spread'].rolling(window=20).std()

# Rolling correlation between the two assets
data['Correlation'] = data[tickers[0]].rolling(window=20).corr(data[tickers[1]])

# Drop NaN values
data = data.dropna().reset_index(drop=True)

# Display the head of the data with new features
data.head()
