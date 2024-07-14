import requests
import pandas as pd


def VisualCrossing(latitude, longitude, date, time, key):

    base_url = 'https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/'
    coordinates = f'{latitude},{longitude}/'
    datetime = f'{date}T{time}:00'
    key = f'?key={key}'
    icon_set = '&iconSet=icons2'
    timezone = '&timezone=UTC'
    include = '&include=current'
    unit_group = '&unitGroup=metric'
    lang = '&lang=id'

    final_url = f'{base_url}{coordinates}{datetime}{key}{icon_set}{timezone}{include}{unit_group}{lang}'

    response = requests.get(final_url)

    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None

    response = response.json()

    keys_to_delete = ['queryCost', 'resolvedAddress', 'address', 'timezone', 'tzoffset', 'days', 'stations']
    for key in keys_to_delete:
        if key in response:
            del response[key]

    df = pd.json_normalize(response)

    columns_to_delete = ['latitude', 'longitude', 'currentConditions.datetime', 'currentConditions.dew', 'currentConditions.snowdepth', 'currentConditions.solarradiation', 'currentConditions.solarenergy', 'currentConditions.stations', 'currentConditions.source', 'currentConditions.severerisk', 'currentConditions.precipprob']
    for col in columns_to_delete:
        if col in df.columns:
            df = df.drop(columns=col)

    df.columns = [col.replace('currentConditions.', '') for col in df.columns]

    return df
