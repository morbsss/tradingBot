import pandas as pd

def process_macro_data(sentiment_df):
    daily_sentiment = sentiment_df.groupby(sentiment_df["timestamp"].dt.date).mean()
    return daily_sentiment