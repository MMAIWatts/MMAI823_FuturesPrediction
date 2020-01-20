import requests
import pandas as pd


def get_weather(datasetid, stationid, datatype, start_date, end_date, token, base_url, locationid=''):
    token = {'token': token}

    params = 'datasetid=' + str(datasetid) + '&' + 'datatypeid' + str(datatype) + '&' + 'stationid=' + str(stationid) + '&' + 'startdate=' + str(
        start_date) + '&' + 'enddate=' + str(end_date) + '&' + 'limit=500' + '&' + 'units=metric'

    r = requests.get(base_url, params=params, headers=token)
    print('Request status code: ' + str(r.status_code))

    try:
        df = pd.DataFrame.from_dict(r.json()['results'])
        print('Successfully retrieved ' + str(len(df['station'].unique())) + ' stations')
        dates = pd.to_datetime(df['date'])
        print('Last date retrieved: ' + str(dates.iloc[-1]))

        return df
    # catch exceptions
    except:
        print('Error converting weather data to DataFrame.')
