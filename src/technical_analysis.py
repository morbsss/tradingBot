import numpy as np
from ta.trend import SMAIndicator, EMAIndicator
from ta.volatility import BollingerBands
from ta.trend import MACD
from ta.momentum import StochasticOscillator, ROCIndicator
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

    df["Volume_MA_10"] = df["Volume"].rolling(window=10).mean()  # 10-day volume moving average
    df["Volume_Ratio"] = df["Volume"] / df["Volume_MA_10"].replace(0, np.nan)  # Avoid division by zero
    df["Volume_MA_20"] = df["Volume"].rolling(window=20).mean()  # 20-day volume moving average

    # Moving Averages
    df["SMA_50"] = SMAIndicator(close=df["Close"], window=50).sma_indicator()
    df["SMA_200"] = SMAIndicator(close=df["Close"], window=200).sma_indicator()
    df["EMA_20"] = EMAIndicator(close=df["Close"], window=20).ema_indicator()
    df["EMA_50"] = EMAIndicator(close=df["Close"], window=50).ema_indicator()

    # Bollinger Bands
    bb = BollingerBands(close=df["Close"], window=20, window_dev=2)
    df["BB_upper"] = bb.bollinger_hband()
    df["BB_lower"] = bb.bollinger_lband()
    df["BB_middle"] = bb.bollinger_mavg()

    # Stochastic Oscillator (14-day, 3-day smoothing)
    stoch = StochasticOscillator(high=df["High"], low=df["Low"], close=df["Close"], window=14, smooth_window=3)
    df["Stochastic_K"] = stoch.stoch()
    df["Stochastic_D"] = stoch.stoch_signal()

    # Rate of Change (14-day)
    df["ROC_14"] = ROCIndicator(close=df["Close"], window=14).roc()

    return df    