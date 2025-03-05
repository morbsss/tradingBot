from backtesting import Backtest, Strategy
import pandas as pd

# Dummy data
data = pd.DataFrame({
    'Open': [100, 101, 102],
    'High': [101, 102, 103],
    'Low': [99, 100, 101],
    'Close': [100, 101, 102],
    'Volume': [1000, 1100, 1200]
}, index=pd.date_range('2023-01-01', periods=3))

class TestStrategy(Strategy):
    def init(self):
        pass
    def next(self):
        if not self.position:
            self.buy()

bt = Backtest(data, TestStrategy, cash=10000, commission=.002)
stats = bt.run()
print(stats)