import requests
import pandas as pd


def is_it_water(latitude, longitude, key):
    url = "https://isitwater-com.p.rapidapi.com/"

    querystring = {"latitude": latitude, "longitude": longitude}

    headers = {
        "x-rapidapi-key": f"{key}",
        "x-rapidapi-host": "isitwater-com.p.rapidapi.com"
    }

    response = requests.get(url, headers=headers, params=querystring)

    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None

    response = response.json()
    response = pd.json_normalize(response)
    response = response.drop(columns=['latitude', 'longitude'])

    if response.water[0] == True:
        icon = 'boat_orange.png'
    else:
        icon = 'person_orange.png'

    response['point_icon'] = icon

    return response
