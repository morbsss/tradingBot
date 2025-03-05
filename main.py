import pandas as pd

from sklearn.preprocessing import StandardScaler

from src.data_download import fetch_historical_data#, fetch_x_sentiment
from src.technical_analysis import calculate_technical_indicators
from src.macro_analysis import process_macro_data
from src.ml_model import train_ml_model, predict, load_model_config
from src.trading_logic import generate_signals, execute_trade
from src.backtesting import compare_strategies

def main():
    config = load_model_config()
    sequence_length = config["LSTM"]["sequence_length"]
    # Fetch data
    df = fetch_historical_data("NVDA", "2015-01-01", "2025-02-26")
    # sentiment_df = fetch_x_sentiment("AAPL stock", max_tweets=100)
    
    # Technical Analysis
    df = calculate_technical_indicators(df)

    # Macro Analysis
    # macro_df = process_macro_data(sentiment_df)
    # df = df.join(macro_df, how="left").fillna(0)
    # df['sentiment'] = 0.5
    
    # Prepare ML features
    features_cols = [
        "MACD", "RSI", "Fib_0.618", "Volume", "Volume_MA_10", "Volume_Ratio", "Volume_MA_20", 
        "SMA_50", "SMA_200", "EMA_20", "EMA_50", "BB_upper", "BB_lower", "BB_middle", 
        "Stochastic_K", "Stochastic_D", "ROC_14"
        ]
    
    target_col = "Target"
    df[target_col] = (df["Close"].shift(-1) > df["Close"]).astype(int)

    df_clean = df[features_cols + [target_col]].dropna()

    # Extract features and target from the cleaned DataFrame
    features = df_clean[features_cols]
    target = df_clean[target_col]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    features_scaled = pd.DataFrame(X_scaled, columns=features.columns, index=features.index)
    
    # Train and predict
    lstm_model = train_ml_model(features[:-sequence_length], target[:-sequence_length], ml_model='LSTM', sequence_length=sequence_length)   
    lstm_predictions = predict(lstm_model, features, ml_model='LSTM', sequence_length=sequence_length)

    xgb_model = train_ml_model(features_scaled, target, ml_model="XGBoost")
    xgb_predictions = predict(xgb_model, features_scaled, ml_model="XGBoost")
    xgb_predictions = xgb_predictions[:len(lstm_predictions)]

    df = df.iloc[sequence_length:sequence_length + len(lstm_predictions)].reset_index(drop=True)

    df = generate_signals(df)

    df["LSTM_Signal"] = lstm_predictions
    df["XGB_Signal"] = xgb_predictions
    # df["LSTM_Signal"] = df["LSTM_Signal"].replace({0: -1, 1: 1})
    # df["XGB_Signal"] = df["XGB_Signal"].replace({0: -1, 1: 1})

    print(df["LSTM_Signal"].value_counts())
    print(df["XGB_Signal"].value_counts())
    # Backtest and compare strategies
    results = compare_strategies(df, cash=10000, commission=0.002)

    # Print sample results for manual inspection
    print("\nSample DataFrame with Signals:")
    print(df[["Close", "Signal", "LSTM_Signal", "XGB_Signal"]].tail())
    
    # print(df[["Close", "MACD", "RSI", "Fib_0.618", "Volume", "Volume_MA_10", "Volume_Ratio", "Volume_MA_20", "SMA_50", "SMA_200", "EMA_20", "EMA_50", "ML_Signal", "Final_Signal"]].tail())

if __name__ == "__main__":
    main()