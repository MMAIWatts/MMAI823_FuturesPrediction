import requests
import pandas as pd


def get_weather(datasetid, stationid, start_date, end_date, token, base_url, locationid=''):
    """
    Fetches weather data from NOAA API.

    :param datasetid: Dataset to fetch from
    :param stationid: Station ID of weather to fetch
    :param start_date: Start date of fetch in ISO format (YYYY-MM-DD)
    :param end_date: End date of fetch in ISO format (YYYY-MM-DD)
    :param token: API Token as dict
    :param base_url: Base url of database to fetch
    :param locationid: optional location ID, station ID overrides
    :return: pandas DataFrame containing results.
    """

    token = {'token': token}

    params = 'datasetid=' + str(datasetid) + '&' + 'stationid=' + str(stationid) + '&' + 'startdate=' + str(
        start_date) + '&' + 'enddate=' + str(end_date) + '&' + 'limit=1000' + '&' + 'units=metric'

    r = requests.get(base_url, params=params, headers=token)
    print('Request status code: ' + str(r.status_code))

    try:
        df = pd.DataFrame.from_dict(r.json()['results'])
        print('Successfully retrieved ' + str(len(df['date'].unique())) + ' days')
        dates = pd.to_datetime(df['date'])
        print('Last date retrieved: ' + str(dates.iloc[-1]))

        return df
    # catch exceptions
    except:
        print('Error converting weather data to DataFrame.')


def series_to_supervised(data, targets, n_in=1, n_out=1, dropnan=True):
    """
    Converts a time series to a supervised learning problem.
    :param data: Input data
    :param targets: Output targets
    :param n_in: Number of input time steps
    :param n_out: Number of output time steps
    :param dropnan: Drop NaN flag
    :return: Aggregated data as a pandas DataFrame
    """

    n_vars = 1 if type(data) is list else data.shape[1]
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(data.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(targets.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars-1, n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars-1, n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg
