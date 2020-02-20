import pandas as pd
from backtesting import Strategy, Backtest
from backtesting.lib import crossover, cross
from backtesting.test import SMA


class SmaCross(Strategy):

    n1 = 3
    n2 = 25

    def init(self):
        self.sma1 = self.I(SMA, self.data.Close, self.n1)
        self.sma2 = self.I(SMA, self.data.Close, self.n2)

    def next(self):
        # if sma1 crosses above sma2, buy the asset
        if crossover(self.sma1, self.sma2):
            self.buy()

        # else if sma1 crosses below sma2, sell it
        elif crossover(self.sma2, self.sma1):
            self.sell()


# local variables
paths = ['data/predictionsH18_1.csv', 'data/predictionsH18_2.csv',
         'data/predictionsH18_3.csv', 'data/predictionsH18_4.csv',
         'data/predictionsH18_5.csv', 'data/predictionsH18_6.csv',
         'data/predictionsH18_7.csv', 'data/predictionsH18_8.csv',
         'data/predictionsH18_9.csv', 'data/predictionsH18_10.csv']
cash = 1000000
commission = 0.0102

for path in paths:
    data = pd.read_csv(path, index_col=0)
    data.index = pd.to_datetime(data.index)
    bt = Backtest(data, SmaCross, cash=cash, commission=commission, trade_on_close=True)
    print(path[15:-4])
    print(bt.run())

    bt.plot()
