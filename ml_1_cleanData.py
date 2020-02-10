import pandas as pd
import numpy as np
from utilities.findFiles import findfiles

# local variables
search_path = 'data/FCOJ'

# find all files in FCOJ directory
files = findfiles(target=search_path, extension='csv')

data = []
for f in files:
    data.append(pd.read_csv(f, index_col=0))

for d in data:
    print(d.info())
