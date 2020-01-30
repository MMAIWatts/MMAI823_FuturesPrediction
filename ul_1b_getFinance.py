import pandas as pd
import numpy as np
from utilities.findFiles import findfiles

# local variables
target = 'data/commodities'
data = []

for file in findfiles(target):
    d = pd.read_csv(file, index_col=0)
    d.drop(labels=['High', 'Low', 'Vol.', 'Change %'], axis=1, inplace=True)
    d['pct'] = d['Price'].pct_change()
    data.append(d)

df = pd.DataFrame()
for i, d in enumerate(data):
    print(d.tail())
    df = pd.merge(df, data[i],  right_index=True)

# df = pd.merge(data[0], data[1], left_index=True, right_index=True)
# df = pd.merge(df, data[2], left_index=True, right_index=True)
# df = pd.merge(df, data[3], left_index=True, right_index=True)
# df = pd.merge(df, data[4], left_index=True, right_index=True)
# df = pd.merge(df, data[5], left_index=True, right_index=True)
# df = pd.merge(df, data[6], left_index=True, right_index=True)
# df = pd.merge(df, data[7], left_index=True, right_index=True)
# df = pd.merge(df, data[8], left_index=True, right_index=True)

print(df.info())
