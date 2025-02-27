import pandas as pd

def generate_signals(df):
    df["Signal"] = 0
    # Buy signal: MACD crosses above signal line and RSI < 70
    df.loc[(df["MACD"] > df["MACD_signal"]) & (df["RSI"] < 70), "Signal"] = 1
    # Sell signal: MACD crosses below signal line and RSI > 30
    df.loc[(df["MACD"] < df["MACD_signal"]) & (df["RSI"] > 30), "Signal"] = -1
    return df

def execute_trade(df, model_predictions):
    df["ML_Signal"] = model_predictions
    df["Final_Signal"] = df["Signal"] * df["ML_Signal"]  # Combine TA and ML signals
    return df