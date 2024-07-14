
import pandas as pd

"""Funkcja dobierająca ikonki do warunków pogodowych"""
def getIcons(conditions, icon, precip, preciptype, snow, cloudcover, datetimeEpoch, sunriseEpoch, sunsetEpoch, wave_direction, winddir):

    def getWeatherIcon():
        if preciptype is not None and 'rain' in preciptype and 'snow' in preciptype:
            return 'rainy_snowy.png'

        elif icon == 'thunder-rain' or icon == 'thunder-showers-night':
            return 'stormy_2.png'
        elif icon == 'thunder-showers-day':
            return 'stormy_sunny.png'

        elif snow is not None and snow > 0:
            if snow > 5:
                return 'snowy_2.png'
            else:
                return 'snowy_1.png'

        elif precip is not None and precip > 0:
            if precip < 2:
                if day and cloudcover < 50:
                    return 'rainy_sunny.png'
                else:
                    return 'rainy_1.png'
            elif 2 <= precip < 5:
                return 'rainy_2.png'
            elif 5 <= precip < 10:
                return 'rainy_3.png'
            elif 10 <= precip < 20:
                return 'rainy_4.png'
            else:
                return 'rainy_5.png'

        else:
            if cloudcover is None:
                return None
            if cloudcover < 25:
                if day:
                    return 'pure_sky_day.png'
                else:
                    return 'pure_sky_night.png'
            elif 25 <= cloudcover < 75:
                if day:
                    return 'cloudy_1_day.png'
                else:
                    return 'cloudy_1_night.png'
            elif 75 <= cloudcover < 90:
                return 'cloudy_2.png'
            else:
                return 'cloudy_3.png'


    def getArrow(direction):
        if direction is None:
            return None

        arrows = ['arrow_N.png', 'arrow_NE.png', 'arrow_E.png', 'arrow_SE.png', 'arrow_S.png', 'arrow_SW.png', 'arrow_W.png', 'arrow_NW.png']
        direction = direction % 360
        idx = round(direction/45) % 8

        return arrows[idx]

    if sunriseEpoch <= datetimeEpoch < sunsetEpoch:
        day = 1
    else:
        day = 0

    wind_arrow = getArrow(winddir)
    wave_arrow = getArrow(wave_direction)

    tmp = conditions.split(", ")

    if len(tmp) != 1:
        weather_icon = getWeatherIcon()
    else:
        if conditions == 'type_1':
            weather_icon = 'snowy_3.png'
        elif conditions == 'type_2':
            weather_icon = 'rainy_3.png'
        elif conditions == 'type_3':
            weather_icon = 'rainy_3.png'
        elif conditions == 'type_4':
            weather_icon = 'rainy_2.png'
        elif conditions == 'type_5':
            weather_icon = 'rainy_3.png'
        elif conditions == 'type_6':
            weather_icon = 'rainy_2.png'
        elif conditions == 'type_7':
            weather_icon = getWeatherIcon()
        elif conditions == 'type_8':
            weather_icon = getWeatherIcon()
        elif conditions == 'type_9':
            weather_icon = 'rainy_snowy.png'
        elif conditions == 'type_10':
            weather_icon = 'rainy_snowy.png'
        elif conditions == 'type_11':
            weather_icon = 'rainy_snowy.png'
        elif conditions == 'type_12':
            weather_icon = getWeatherIcon()
        elif conditions == 'type_13':
            weather_icon = 'rainy_snowy.png'
        elif conditions == 'type_14':
            weather_icon = 'rainy_snowy.png'
        elif conditions == 'type_15':
            weather_icon = getWeatherIcon()
        elif conditions == 'type_16':
            weather_icon = 'snowy_1.png'
        elif conditions == 'type_17':
            weather_icon = 'snowy_3.png'
        elif conditions == 'type_18':
            weather_icon = 'lightning.png'
        elif conditions == 'type_19':
            weather_icon = getWeatherIcon()
        elif conditions == 'type_20':
            weather_icon = 'rainy_2.png'
        elif conditions == 'type_21':
            weather_icon = 'rainy_3.png'
        elif conditions == 'type_22':
            weather_icon = 'rainy_snowy.png'
        elif conditions == 'type_23':
            weather_icon = 'rainy_snowy.png'
        elif conditions == 'type_24':
            weather_icon = 'rainy_4.png'
        elif conditions == 'type_25':
            weather_icon = 'rainy_5.png'
        elif conditions == 'type_26':
            weather_icon = 'rainy_1.png'
        elif conditions == 'type_27':
            weather_icon = 'cloudy_2.png'
        elif conditions == 'type_28':
            weather_icon = 'cloudy_3.png'
        elif conditions == 'type_29':
            weather_icon = getWeatherIcon()
        elif conditions == 'type_30':
            weather_icon = getWeatherIcon()
        elif conditions == 'type_31':
            weather_icon = 'snowy_3.png'
        elif conditions == 'type_32':
            weather_icon = 'rainy_snowy.png'
        elif conditions == 'type_33':
            weather_icon = 'snowy_1.png'
        elif conditions == 'type_34':
            weather_icon = 'snowy_2.png'
        elif conditions == 'type_35':
            weather_icon = 'snowy_1.png'
        elif conditions == 'type_36':
            weather_icon = 'wind_1.png'
        elif conditions == 'type_37':
            weather_icon = 'stormy_3.png'
        elif conditions == 'type_38':
            weather_icon = 'stormy_1.png'
        elif conditions == 'type_39':
            weather_icon = getWeatherIcon()
        elif conditions == 'type_40':
            weather_icon = 'rainy_5.png'
        elif conditions == 'type_41':
            weather_icon = 'cloudy_3.png'
        elif conditions == 'type_42':
            weather_icon = getWeatherIcon()
        elif conditions == 'type_43':
            weather_icon = getWeatherIcon()
        else:
            weather_icon = getWeatherIcon()

    descriptions = ['Zawieje lub zamiecie śnieżne', 'Mżawka', 'Silna mżawka', 'Słaba mżawka', 'Silna mżawka/deszcz', 'Słaba mżawka/deszcz', 'Burza pyłowa lub piaskowa', 'Mgła', 'Marznąca mżawka/marznący deszcz', 'Silna marznąca mżawka/marznący deszcz', 'Słaba marznąca mżawka/marznący deszcz', 'Marznąca mgła', 'Silny marznący deszcz', 'Słaby marznący deszcz', 'Lej kondensacyjny/Tornado', 'Przelotne opady gradu', 'Lód', 'Błyskawica bez grzmotu', 'Zamglenie', 'Opady w pobliżu', 'Deszcz', 'Silny deszcz i śnieg', 'Słaby deszcz i śnieg', 'Opady deszczu', 'Silny deszcz', 'Słaby deszcz', 'Zachmurzenie malejące', 'Zachmurzenie rosnące', 'Niebo bez zmian', 'Smog lub zmętnienie', 'Śnieg', 'Opady śniegu i deszczu', 'Opady śniegu', 'Silne opady śniegu', 'Słabe opady deszczu', 'Szkwały', 'Burza z piorunami', 'Burza bez opadów', 'Pył diamentowy', 'Grad', 'Pochmurnie', 'Częściowe zachmurzenie', 'Bezchmurnie']
    types = ['type_1', 'type_2', 'type_3', 'type_4', 'type_5', 'type_6', 'type_7', 'type_8', 'type_9', 'type_10', 'type_11', 'type_12', 'type_13', 'type_14', 'type_15', 'type_16', 'type_17', 'type_18', 'type_19', 'type_20', 'type_21', 'type_22', 'type_23', 'type_24', 'type_25', 'type_26', 'type_27', 'type_28', 'type_29', 'type_30', 'type_31', 'type_32', 'type_33', 'type_34', 'type_35', 'type_36', 'type_37', 'type_38', 'type_39', 'type_40', 'type_41', 'type_42', 'type_43']

    description = conditions
    for i in range(0, 43):
        if f'{types[43-i-1]}' in description:
            description = description.replace(f'{types[43-i-1]}', f'{descriptions[43-i-1]}')

    data = {
        'wind_arrow': [wind_arrow],
        'wave_arrow': [wave_arrow],
        'weather_icon': [weather_icon],
        'description': [description],
        'day': [day]
    }

    df = pd.DataFrame(data)

    return df
