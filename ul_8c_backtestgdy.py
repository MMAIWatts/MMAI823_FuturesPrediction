import pandas as pd
from backtesting import Strategy, Backtest
from backtesting.lib import crossover, cross
from backtesting.test import SMA

actualpath = 'data/FCOJ/gdy00_backtest.csv'


class SmaCross19(Strategy):

    n1 = 1
    n2 = 15

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
cash = 1000000
commission = 0.0102

data = pd.read_csv(actualpath, index_col=0)
data.index = pd.to_datetime(data.index)
bt = Backtest(data, SmaCross19, cash=cash, commission=commission, trade_on_close=True)
print(bt.run())

bt.plot()
