import pandas as pd
import openmeteo_requests
import requests_cache
from retry_requests import retry


def OpenMeteo(latitude, longitude, date, time):

    hour, minute = time.split(':')
    hour, minute = int(hour), int(minute)
    if minute > 30 and hour != 23:
        hour += 1

    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    url = "https://marine-api.open-meteo.com/v1/marine"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": ["wave_height", "wave_direction", "wave_period", "wind_wave_height", "wind_wave_direction", "wind_wave_period", "swell_wave_height", "swell_wave_direction", "swell_wave_period"],
        "timezone": "GMT",
        "start_date": date,
        "end_date": date,
        "models": "best_match"
      }
    responses = openmeteo.weather_api(url, params=params)

    response = responses[0]

    hourly = response.Hourly()
    hourly_wave_height = hourly.Variables(0).ValuesAsNumpy()
    hourly_wave_direction = hourly.Variables(1).ValuesAsNumpy()
    hourly_wave_period = hourly.Variables(2).ValuesAsNumpy()
    hourly_wind_wave_height = hourly.Variables(3).ValuesAsNumpy()
    hourly_wind_wave_direction = hourly.Variables(4).ValuesAsNumpy()
    hourly_wind_wave_period = hourly.Variables(5).ValuesAsNumpy()
    hourly_swell_wave_height = hourly.Variables(6).ValuesAsNumpy()
    hourly_swell_wave_direction = hourly.Variables(7).ValuesAsNumpy()
    hourly_swell_wave_period = hourly.Variables(8).ValuesAsNumpy()

    hourly_data = {"date": pd.date_range(
        start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
        end = pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
        freq = pd.Timedelta(seconds = hourly.Interval()),
        inclusive = "left"
    )}

    hourly_data["wave_height"] = hourly_wave_height
    hourly_data["wave_direction"] = hourly_wave_direction
    hourly_data["wave_period"] = hourly_wave_period
    hourly_data["wind_wave_height"] = hourly_wind_wave_height
    hourly_data["wind_wave_direction"] = hourly_wind_wave_direction
    hourly_data["wind_wave_period"] = hourly_wind_wave_period
    hourly_data["swell_wave_height"] = hourly_swell_wave_height
    hourly_data["swell_wave_direction"] = hourly_swell_wave_direction
    hourly_data["swell_wave_period"] = hourly_swell_wave_period

    df = pd.DataFrame(data=hourly_data)
    df = df.iloc[[hour]].reset_index(drop=True)
    df = df.drop(columns=['date'])

    # jeśli chcemy więcej danych na temat fal z wiatru i martwych fal to zakomentować tę linijkę
    df = df.drop(columns=['wind_wave_height', 'wind_wave_direction', 'wind_wave_period', 'swell_wave_height', 'swell_wave_direction', 'swell_wave_period'])

    return df
