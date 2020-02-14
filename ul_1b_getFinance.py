import pandas as pd
import numpy as np
from utilities.findFiles import findfiles
from matplotlib import pyplot as plt

# set Pandas options
pd.set_option('display.width', 2000)
pd.set_option('display.max_columns', 500)

# local variables
contracts = ['H', 'K', 'N', 'U', 'X']

dataframes = []
for c in contracts:
    target = 'data/FCOJ/' + c
    data = []

    paths = findfiles(target, extension='.csv')
    df = pd.read_csv(paths[0], index_col=0, skipfooter=1, engine='python')
    df.drop(['Open', 'High', 'Low', 'Change', 'Volume', 'Open Int'], axis=1, inplace=True)
    df.index = pd.to_datetime(df.index)
    df.iloc[:, 0] = df['Last'].pct_change()

    col_labels = ['07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19']
    df.columns = [col_labels[0]]

    for i in np.arange(len(paths) - 2):
        d = pd.read_csv(paths[i + 1], index_col=0, skipfooter=1, engine='python')
        d.drop(['Open', 'High', 'Low', 'Change', 'Volume', 'Open Int'], axis=1, inplace=True)
        d.index = pd.to_datetime(d.index)
        d.iloc[:, 0] = d['Last'].pct_change()
        d.columns = [col_labels[i + 1]]
        df = df.merge(d, how='outer', left_index=True, right_index=True)

    df = df.add_prefix(c)
    print(df.info())
    dataframes.append(df)

# merge all dataframes
df_merged = pd.concat(dataframes, axis=1)
print(df_merged.info())
df_merged.to_csv('out/fcoj_merged.csv')

