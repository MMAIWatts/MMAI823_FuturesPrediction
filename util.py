import requests
import pandas as pd


def get_weather(locationid, datasetid, begin_date, end_date, myToken, base_url):
    token = {'token': myToken}

    params = 'datasetid=' + str(datasetid) + '&' + 'locationid=' + str(locationid) + '&' + 'startdate=' + str(
        begin_date) + '&' + 'enddate=' + str(end_date) + '&' + 'limit=25' + '&' + 'units=standard'

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
