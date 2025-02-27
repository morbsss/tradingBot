import pandas as pd
from src.data_download import fetch_historical_data#, fetch_x_sentiment
from src.technical_analysis import calculate_technical_indicators
from src.macro_analysis import process_macro_data
from src.ml_model import train_ml_model, predict
from src.trading_logic import generate_signals, execute_trade

def main():
    # Fetch data
    df = fetch_historical_data("AAPL", "2024-01-01", "2025-02-26")
    # sentiment_df = fetch_x_sentiment("AAPL stock", max_tweets=100)
    
    # Technical Analysis
    df = calculate_technical_indicators(df)
    
    # Macro Analysis
    # macro_df = process_macro_data(sentiment_df)
    # df = df.join(macro_df, how="left").fillna(0)
    # df['sentiment'] = 0.5
    
    # Prepare ML features
    features = df[["MACD", "RSI", "Fib_0.618"]]
    target = (df["Close"].shift(-1) > df["Close"]).astype(int)  # Predict price increase
    
    # Train and predict
    model = train_ml_model(features[:-1], target[:-1])
    predictions = predict(model, features)
    
    # Trading logic
    df = generate_signals(df)
    df = execute_trade(df, predictions)
    
    print(df[["Close", "MACD", "RSI", "Fib_0.618", "ML_Signal", "Final_Signal"]].tail())

if __name__ == "__main__":
    main()