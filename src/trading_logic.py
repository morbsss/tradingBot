import pandas as pd

def generate_signals(df):
    df["Signal"] = 0
    
    # Use Fibonacci 61.8% level as default support/resistance
    fib_level = 0.618  # Default to 0.618
    fib_col = f"Fib_{fib_level}"
    
    if fib_col not in df.columns:
        raise ValueError(f"Fibonacci level column {fib_col} not found in DataFrame")
    
    # Define thresholds and tolerances (configurable or hardcoded for simplicity)
    rsi_overbought = 70  # RSI > 70 indicates overbought
    rsi_oversold = 30    # RSI < 30 indicates oversold
    volume_threshold = 1.5  # Volume ratio > 1.5 indicates high activity
    
    # Buy Signal (Bullish Conditions)
    # 1. MACD bullish crossover (MACD > MACD_signal)
    # 2. RSI not overbought (RSI < 70)
    # 3. Price near or below Fibonacci support (Close <= Fib_0.618 * 1.01)
    # 4. Volume confirmation (Volume_Ratio > 1 or Volume > Volume_MA_10)
    # 5. Moving averages confirm bullish trend (e.g., EMA_20 > EMA_50 and SMA_50 > SMA_200)
    df.loc[
        (df["MACD"] > df["MACD_signal"]) &  # Bullish MACD crossover
        (df["RSI"] < rsi_overbought) &       # Not overbought
        (df["Close"] <= df[fib_col] * 1.01) &  # Near Fibonacci support (1% tolerance)
        ((df["Volume_Ratio"] > 1.0) | (df["Volume"] > df["Volume_MA_10"])) &  # Volume confirmation
        (df["EMA_20"] > df["EMA_50"]) &      # Short-term bullish trend
        (df["SMA_50"] > df["SMA_200"]),      # Long-term bullish trend
        "Signal"
    ] = 1
    
    # Sell Signal (Bearish Conditions)
    # 1. MACD bearish crossover (MACD < MACD_signal)
    # 2. RSI not oversold (RSI > 30)
    # 3. Price near or above Fibonacci resistance (Close >= Fib_0.618 * 0.99)
    # 4. Volume confirmation (Volume_Ratio > 1.5 or Volume > Volume_MA_20)
    # 5. Moving averages confirm bearish trend (e.g., EMA_20 < EMA_50 and SMA_50 < SMA_200)
    df.loc[
        (df["MACD"] < df["MACD_signal"]) &  # Bearish MACD crossover
        (df["RSI"] > rsi_oversold) &         # Not oversold
        (df["Close"] >= df[fib_col] * 0.99) &  # Near Fibonacci resistance (1% tolerance)
        ((df["Volume_Ratio"] > volume_threshold) | (df["Volume"] > df["Volume_MA_20"])) &  # Volume confirmation
        (df["EMA_20"] < df["EMA_50"]) &      # Short-term bearish trend
        (df["SMA_50"] < df["SMA_200"]),      # Long-term bearish trend
        "Signal"
    ] = -1
    
    return df

def execute_trade(df, model_predictions):
    df["ML_Signal"] = model_predictions
    df["Final_Signal"] = df["Signal"] * df["ML_Signal"]
    return df