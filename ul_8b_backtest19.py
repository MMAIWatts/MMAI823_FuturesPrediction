import pandas as pd
from backtesting import Strategy, Backtest
from backtesting.lib import crossover, cross
from backtesting.test import SMA

paths = ['data/predictionsH19_1.csv', 'data/predictionsH19_2.csv',
         'data/predictionsH19_3.csv', 'data/predictionsH19_4.csv',
         'data/predictionsH19_5.csv', 'data/predictionsH19_6.csv',
         'data/predictionsH19_7.csv', 'data/predictionsH19_update.csv',
         'data/predictionsH19_9.csv', 'data/predictionsH19_10.csv']
actualpath = 'data/spotpriceH19.csv'


class SmaCross19(Strategy):

    n1 = 2
    n2 = 40
    n3 = 15

    def init(self):
        spotprice = pd.read_csv(paths[7], index_col=0)
        spotprice.index = pd.to_datetime(spotprice.index)
        self.sma1 = self.I(SMA, spotprice.Close, self.n1)
        self.sma2 = self.I(SMA, spotprice.Close, self.n2)
        self.sma3 = self.I(SMA, self.data.Close, self.n3)

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
results = bt.run()
print(results)
df= results._trade_data
print(df.head())
# df.to_csv('out/h19_trading_results.csv')

bt.plot()
