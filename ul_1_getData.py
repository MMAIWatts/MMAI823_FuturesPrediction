import datetime
import numpy as np
import pandas as pd
from util import get_weather

# NOAA API Token

myToken = 'oGjWvAZgOEWogyaRmrdJxJgyCnWwTMFq'

# get datetime of last year
lastyear = datetime.datetime.now() - datetime.timedelta(weeks=2)

# set start and end dates, currently set to 1 day's data for testing
begin_date = lastyear.strftime("%Y-%m-%d")
end_date = lastyear.strftime("%Y-%m-%d")

# location id for the locations of interest
locationid = 'FIPS:38'
datasetid = 'GHCND'

# urls
base_url_data = 'https://www.ncdc.noaa.gov/cdo-web/api/v2/data'
base_url_stations = 'https://www.ncdc.noaa.gov/cdo-web/api/v2/stations'

df_weather = get_weather(locationid, datasetid, begin_date, end_date, myToken, base_url_data)

print(df_weather.head())
