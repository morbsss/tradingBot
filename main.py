import pandas as pd
from src.data_download import fetch_historical_data#, fetch_x_sentiment
from src.technical_analysis import calculate_technical_indicators
from src.macro_analysis import process_macro_data
from src.ml_model import train_ml_model, predict
from src.trading_logic import generate_signals, execute_trade

def main():
    sequence_length = 20
    # Fetch data
    df = fetch_historical_data("ETH-USD", "2015-01-01", "2025-02-26")
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
    ltsm_model = train_ml_model(features[:-1], target[:-1], ml_model='LSTM', sequence_length=sequence_length)   
    ltsm_predictions = predict(ltsm_model, features, ml_model='LSTM', sequence_length=sequence_length)

    xgb_model = train_ml_model(features, target, ml_model="XGBoost")
    importance = xgb_model.feature_importances_
    feature_names = features.columns
    for name, imp in zip(feature_names, importance):
        print(f"Feature: {name}, Importance: {imp:.4f}")
    xgb_predictions = predict(xgb_model, features, ml_model="XGBoost")
    
    # Trading logic for LSTM
    df_lstm = df.copy()
    df_lstm = df_lstm.iloc[sequence_length:sequence_length + len(ltsm_predictions)].reset_index(drop=True)
    df_lstm = generate_signals(df_lstm)
    df_lstm = execute_trade(df_lstm, ltsm_predictions)

    df_xgb = df.copy()
    df_xgb = generate_signals(df_xgb)
    df_xgb = execute_trade(df_xgb, xgb_predictions)
    
    print(df_lstm[["Close", "MACD", "RSI", "Fib_0.618", "ML_Signal", "Final_Signal"]].tail())
    print(df_xgb[["Close", "MACD", "RSI", "Fib_0.618", "ML_Signal", "Final_Signal"]].tail())

if __name__ == "__main__":
    main()