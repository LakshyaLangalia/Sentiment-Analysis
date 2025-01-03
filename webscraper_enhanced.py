# Importing the necessary libraries

from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import yfinance as yf
from transformers import pipeline

# Getting the news from the website and getting it ready for sentiment analysis/processing
def fetch_news(ticker):
    # Base URL which is then appended by the ticker
    # Sends a request to the wesbite, with the appropriate headers to ensure that the website accepts the request
    url = f'https://finviz.com/quote.ashx?t={ticker}'
    req = Request(url, headers={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    })

    # opens the URL and using BeautifulSoup, the html text is processed and made into an analyzable format
    response = urlopen(req)
    html = BeautifulSoup(response, 'html.parser')

    # extracts the news table from the rest of the HTML so that only it is processed.
    news_table = html.find(id='news-table')

    parsed_data = []

    # extracts all table rows in the HTML and iterates through them, grabbing the
    # title, date, and time, and then adding that to the parsed data list as a news article ready for
    # sentiment processing. 
    for row in news_table.findAll('tr'):
        title = row.a.text
        date_data = row.td.text.split()
        date = date_data[0] if len(date_data) > 1 else 'Today' # Today is used as a placeholder
        time = date_data[-1]
        parsed_data.append([ticker, date, time, title])

    # constructs a DataFrame with all articles, for easier processing
    df = pd.DataFrame(parsed_data, columns=['ticker', 'date', 'time', 'title'])
    df['date'] = df['date'].replace('Today', datetime.now().strftime('%b-%d-%y'))
    df['date'] = pd.to_datetime(df['date']).dt.date
    return df

# Analyzes sentiment using finbert, which is a financial text processing model specifically
def analyze_sentiment(df):
    sentiment_model = pipeline('text-classification', model='ProsusAI/finbert')

    # assigns a score to every title in the dataframe by applying the sentiment_model to each headline
    # the model will then predict the sentiment without having to parse through the entire article
    # the model will return a list of dicts for each prediction, as well as a confidence value 
    df['score'] = df['title'].apply(lambda title: sentiment_model(title)[0]['score'])

    # finds the average sentiment score for each day and returns a Series containing that info
    return df.groupby('date')['score'].mean()

# Gets stock price data from the specified start and end date
def fetch_stock_data(ticker, start_date, end_date):
    stock = yf.Ticker(ticker).history(start=start_date, end=end_date)['Close']
    return stock

# Plots the stock price vs sentiment
def plot_sentiment_and_price(sentiment, price, ticker):

    # calculates the exponential moving average, which gives more weight to recent information
    # does so for the price and sentiment columns
    sentiment_ema = sentiment.ewm(span=5, adjust=False).mean()
    price_ema = price.ewm(span=5, adjust=False).mean()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax1.plot(sentiment.index, sentiment.values, label='Sentiment', color='blue', linewidth=1.5)
    ax1.plot(sentiment_ema.index, sentiment_ema.values, label='EMA (5 days)', color='orange', linewidth=1.5)
    ax1.set_title('Stock Sentiment vs Time')
    ax1.set_ylabel('Sentiment')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(price.index, price.values, label='Stock Price', color='green', linewidth=1.5)
    ax2.plot(price_ema.index, price_ema.values, label='EMA (5 Days)', color='purple', linewidth=1.5)
    ax2.set_title(f'{ticker} Stock Price')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Price ($)')
    ax2.legend()
    ax2.grid(True)

    plt.show()

    corr= sentiment.reindex(price.index).dropna().corr(price.dropna())
    print(f"Correlation coefficient: {corr}")

ticker = 'AOS'
news_df = fetch_news(ticker)
sentiment = analyze_sentiment(news_df)

start_date = sentiment.index.min()
end_date = sentiment.index.max()
price = fetch_stock_data(ticker, start_date, end_date)

plot_sentiment_and_price(sentiment, price, ticker)
