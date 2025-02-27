import pandas as pd
from ta.trend import MACD
from ta.momentum import RSIIndicator

def calculate_technical_indicators(df):
    # MACD
    macd = MACD(df["Close"], window_fast=12, window_slow=26, window_sign=9)
    df["MACD"] = macd.macd()
    df["MACD_signal"] = macd.macd_signal()
    
    # RSI
    rsi = RSIIndicator(df["Close"], window=14)
    df["RSI"] = rsi.rsi()
    
    # Fibonacci Retracement
    df["Fib_0.618"] = df["High"].max() - (df["High"].max() - df["Low"].min()) * 0.618
    return df