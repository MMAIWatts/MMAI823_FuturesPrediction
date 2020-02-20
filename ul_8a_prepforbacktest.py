import pandas as pd


# set Pandas options
pd.set_option('display.width', 2000)
pd.set_option('display.max_columns', 500)

# set local variables
pathpred18 = 'out/predictions_mse_H18.csv'
xpath18 = 'out/supervised_data/H18_supervised.csv'
pathpred19 = 'out/predictions_mse_H19.csv'
xpath19 = 'out/supervised_data/H19_supervised.csv'

# read data and set dateindex from initial data
pred18 = pd.read_csv(pathpred18, index_col=0)
pred19 = pd.read_csv(pathpred19, index_col=0)
pred18.index = pd.to_datetime(pd.read_csv(xpath18, index_col=0, skiprows=2).index)
pred19.index = pd.to_datetime(pd.read_csv(xpath19, index_col=0, skiprows=2).index)

# set initial prices from historic data
startprice18 = 123.85
startprice19 = 145.60





