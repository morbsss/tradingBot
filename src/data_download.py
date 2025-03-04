import pandas as pd
import yfinance as yf

def fetch_historical_data(symbol, start_date, end_date):
    stock = yf.Ticker(symbol)
    df = stock.history(start=start_date, end=end_date)
    return df

# def fetch_x_sentiment(query, max_tweets=100):
#     headers = {
#         "Authorization": "Bearer " + config["x_api"]["bearer_token"]
#     }
#     url = "https://api.x.com/2/tweets/search/recent"

#     params = {
#         "query": query,
#         "max_results": max_tweets
#     }

#     x_dict = {}    
#     response = requests.get(url, headers=headers, params=params)
#     # print(response)
#     if response .status_code == 200:
#         tweets = response.json()
#         print(tweets)
#         for tweet in tweets.get("data", []):
#             sentiment_scores = analyze_sentiment(tweet["text"])
#             x_dict[tweet["created_at"]] = sentiment_scores

#     print(x_dict)



    # auth = tweepy.OAuthHandler(config["x_api"]["consumer_key"], config["x_api"]["consumer_secret"])
    # auth.set_access_token(config["x_api"]["access_token"], config["x_api"]["access_token_secret"])
    # api = tweepy.API(auth)
    
    # tweets = tweepy.Cursor(api.search_tweets, q=query, lang="en").items(max_tweets)
    # sentiment_scores = [analyze_sentiment(tweet.text) for tweet in tweets]
    # return pd.DataFrame({"timestamp": [t.created_at for t in tweets], "sentiment": sentiment_scores})

# def analyze_sentiment(text):
#     # Placeholder: Use a proper sentiment analysis model (e.g., VADER or transformers)
#     return 0.5  # Neutral score for demo