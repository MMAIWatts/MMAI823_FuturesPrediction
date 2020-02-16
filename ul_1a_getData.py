import datetime
import numpy as np
import pandas as pd
from util import get_weather
import fileinput

# NOAA API Token
api_token_path = 'data/noaa_api_token.txt'
myToken = fileinput.FileInput(api_token_path).readline()
# Path to location list
station_path = 'data/city_codes.csv'

# import list of locations
df_stations = pd.read_csv(station_path)
stations = df_stations.StationID
print('Fetching the following stations...')
print(df_stations)

# Station ID for the locations of interest
datasetid = 'GHCND'

# urls
base_url_data = 'https://www.ncdc.noaa.gov/cdo-web/api/v2/data'

years = [2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019]
for y in years:
    begin_date = datetime.date(year=y, month=1, day=1)
    df_weather = pd.DataFrame()
    # fetch data from NOAA and store in csv
    for s in stations:
        print(s)
        for i in np.arange(0, 13):
            start = begin_date + datetime.timedelta(weeks=int(i * 4))
            end = start + datetime.timedelta(weeks=4)
            d = get_weather(stationid=s, datasetid=datasetid, start_date=start.strftime('%Y-%m-%d'),
                            end_date=end.strftime('%Y-%m-%d'), token=myToken, base_url=base_url_data)
            df_weather = df_weather.append(d, ignore_index=True)
    df_weather.to_csv('out/' + str(begin_date.year) + '_raw_weather_data.csv')
