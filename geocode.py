
import requests
import pandas as pd


def get_geocode_data(api_key, latitude, longitude, date, time, language="pl"):
    url = f"https://api.opencagedata.com/geocode/v1/json"
    params = {
        'key': api_key,
        'q': f"{latitude},{longitude}",
        'pretty': 1,
        'no_annotations': 1,
        'language': language
    }

    response = requests.get(url, params=params)

    if response.status_code == 200:
        response = response.json()
        if 'results' in response and len(response['results']) > 0:
            formatted_address = response['results'][0]['formatted']
            df = pd.DataFrame([{
                'latitude': float(latitude),
                'longitude': float(longitude),
                'location': formatted_address,
                'date': date,
                'time': time
            }])
            return df
    else:
        return {"error": f"Request failed with status code {response.status_code}"}
