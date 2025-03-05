import pandas as pd
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

class TradingStrategy(Strategy):
    """
    Backtesting strategy using rule-based, LSTM, or XGBoost signals.
    """
    # Define class-level parameters
    n1 = 12  # MACD fast period (for reference, not used directly here)
    n2 = 26  # MACD slow period
    n3 = 9   # MACD signal period
    signal_type = "Signal"  # Default signal type as a class variable
    
    def init(self):
        # Load the appropriate signal based on signal_type
        self.signals = self.data[self.signal_type]  # Dynamically select signal column
        self.close = self.data.Close
    
    def next(self):
        # Use only the selected signal type for trading logic
        if self.signals[-1] == 1:  # Buy signal
            # if not self.position:  # If no position, go long
            self.buy()
        elif self.signals[-1] == -1:  # Sell signal
            # if self.position:  # If long position, close it
            self.position.close()

def backtest_strategy(df, signal_column, cash=10000, commission=0.002):
    """
    Backtest a trading strategy using the specified signal column.
    
    Args:
        df (pd.DataFrame): DataFrame with 'Close' and signal column.
        signal_column (str): Column name for signals (e.g., 'Signal', 'LSTM_Signal', 'XGB_Signal').
        cash (float): Starting cash for backtest.
        commission (float): Commission per trade (as a fraction of trade value).
    
    Returns:
        Backtest: Backtest object with results.
    """
    # Ensure signals are numeric and map to -1, 0, 1 if needed
    df[signal_column] = df[signal_column].astype(int)
    
    # Prepare data for backtesting
    bt_df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    bt_df[signal_column] = df[signal_column]
    
    # Create and run backtest with signal_type parameter
    bt = Backtest(bt_df, TradingStrategy,
                  cash=cash, commission=commission,
                  exclusive_orders=True,  # Prevents overlapping trades
                  trade_on_close=True)    # Trades on candle close for realism
    
    stats = bt.run(signal_type=signal_column)  # Pass signal column as a parameter
    return bt, stats

def compare_strategies(df, cash=10000, commission=0.002):
    """
    Compare rule-based, LSTM, and XGBoost strategies.
    
    Args:
        df (pd.DataFrame): DataFrame with all signals ('Signal', 'LSTM_Signal', 'XGB_Signal').
        cash (float): Starting cash for backtest.
        commission (float): Commission per trade.
    
    Returns:
        dict: Dictionary of backtest stats for each strategy.
    """
    strategies = {
        "Rule_Based": "Signal",
        "LSTM": "LSTM_Signal",
        "XGBoost": "XGB_Signal"
    }
    
    results = {}
    for name, signal_col in strategies.items():
        bt, stats = backtest_strategy(df, signal_col, cash, commission)
        results[name] = stats
        print(f"\n{name} Strategy Results:")
        print(stats)
        bt.plot(filename=f"{name}_equity_curve.html")  # Save plot to file
    
    return results

# Example usage (uncomment and adjust with your data)
# df = pd.read_csv("your_data.csv")
# results = compare_strategies(df)