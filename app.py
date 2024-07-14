from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import numpy as np
import math
from matplotlib.dates import DateFormatter, AutoDateLocator
import ast
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from datetime import datetime
from scipy.interpolate import interp1d
import os
import threading
import time
from geocode import get_geocode_data
from isitwater import is_it_water
from openmeteo import OpenMeteo
from visualcrossing import VisualCrossing
import getkeys
from geticons import getIcons
from animation import getAnimation
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.gridspec import GridSpec
from moviepy.editor import ImageSequenceClip

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
file_content = None

with open('dane.txt', 'r', encoding='utf-8') as file:
    data = ast.literal_eval(file.read())


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/page1')
def page1():
    global file_content
    global data
    data_2 = pd.DataFrame(data)

    if file_content is not None:
        result = file_content.to_html(classes='data', header="true", index=False)
    else:
        result = data_2.to_html(classes='data', header="true", index=False)
    return render_template('page1.html', file_content=result)


@app.route('/page2')
def page2():
    global file_content
    global data
    if file_content is None:
        df = pd.DataFrame(data)
        file_content = df

    df_new = ''
    selected_columns = ["conditions", "icon", "precip", "preciptype", "snow", "cloudcover", "datetimeEpoch",
                        "sunriseEpoch", "sunsetEpoch", "wave_direction", "winddir"]
    if file_content is not None:
        list_of_lists = file_content[selected_columns].values.tolist()
        list_of_lists = [[None if item is None or item == 'null' or item == 'NaN' or (isinstance(item, float) and math.isnan(item)) else item for item in sublist] for sublist in list_of_lists]
        temp_dfs = []

        for i in list_of_lists:
            result = getIcons(i[0], i[1], i[2], i[3], i[4], i[5], i[6], i[7], i[8], i[9], i[10])
            temp_dfs.append(result)
        concated = pd.concat(temp_dfs, ignore_index=True)
        new_ = pd.concat([file_content, concated], axis=1)
        selected_columns = ['latitude', 'longitude', 'point_icon', 'water', 'location', 'date', 'time', 'wind_arrow',
                            'windspeed', 'wave_arrow', 'wave_height', 'temp', 'weather_icon', 'description']

        df_new = new_[selected_columns].copy()
        df_new['wave_arrow'] = df_new['wave_arrow'].apply(lambda x: None if pd.isnull(x) else x)
        threading.Thread(target=long_running_task, args=(df_new,)).start()
        os.makedirs('frames', exist_ok=True)

        df_new['wave_height'] = df_new['wave_height'].round(1)

        # Obliczanie zakresów geograficznych
        min_lat = df_new['latitude'].min()
        max_lat = df_new['latitude'].max()
        min_lon = df_new['longitude'].min()
        max_lon = df_new['longitude'].max()

        lat_diff = max_lat - min_lat
        lon_diff = max_lon - min_lon

        buffer_lat = lat_diff * 0.05
        buffer_lon = lon_diff * 0.05

        if lat_diff > lon_diff:
            buffer_lon = (lat_diff - lon_diff) / 2
        elif lon_diff > lat_diff:
            buffer_lat = (lon_diff - lat_diff) / 2

        min_lat -= buffer_lat
        max_lat += buffer_lat
        min_lon -= buffer_lon
        max_lon += buffer_lon

        if min_lat < -90:
            min_lat = -90
        if max_lat > 90:
            max_lat = 90
        if min_lon < -180:
            min_lon = -180
        if max_lon > 180:
            max_lon = 180

        if max_lon - min_lon >= 180:
            min_lat = -90
            max_lat = 90

        lat_diff = max_lat - min_lat
        lon_diff = max_lon - min_lon

        max_figsize = 10
        aspect_ratio = lon_diff / lat_diff
        if aspect_ratio > 1:
            figsize = (max_figsize, max_figsize / aspect_ratio + 2)
        else:
            figsize = (max_figsize * aspect_ratio, max_figsize + 2)

        # Tworzenie figury i mapy z odpowiednim kadrowaniem
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(3, 1, height_ratios=[0.5, 8, 2], hspace=0.1, top=0.95, bottom=0.05, left=0.05, right=0.95)
        text_ax = fig.add_subplot(gs[0])
        ax = fig.add_subplot(gs[1], projection=ccrs.PlateCarree())
        desc_ax = fig.add_subplot(gs[2])

        text_ax.axis('off')
        desc_ax.axis('off')
        ax.set_extent([min_lon, max_lon, max_lat, min_lat])

        if lon_diff > 60 and lat_diff > 60:
            ax.stock_img()

        ax.add_feature(cfeature.LAND)
        ax.add_feature(cfeature.OCEAN)
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.LAKES)
        ax.add_feature(cfeature.RIVERS)
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        ax.gridlines(draw_labels=True)

        annotation_boxes = []
        captions = []
        filenames = []

        for i, row in df_new.iterrows():
            # Usuwanie poprzednich ikon
            for ab in annotation_boxes:
                ab.remove()
            annotation_boxes = []

            for caption in captions:
                caption.remove()
            captions = []

            ax.scatter(row['longitude'], row['latitude'], color='orange', s=10, transform=ccrs.PlateCarree())

            icon_img = mpimg.imread(f"icons/{row['point_icon']}")
            imagebox = OffsetImage(icon_img, zoom=0.07)
            ab = AnnotationBbox(imagebox, (row['longitude'], row['latitude']), frameon=False,
                                transform=ccrs.PlateCarree())
            ax.add_artist(ab)
            annotation_boxes.append(ab)

            text1 = text_ax.text(0.05, 0.75, f"Czas:  {row['date']}   {row['time']}", ha='left',
                                 va='bottom', fontsize=10, weight='bold', transform=text_ax.transAxes)
            text2 = text_ax.text(0.05, 0.6, f"Miejsce:  {row['location']}  [{row['latitude']}, {row['longitude']}]",
                                 ha='left', va='top', fontsize=10, weight='bold', transform=text_ax.transAxes)

            captions.append(text1)
            captions.append(text2)

            # Dodanie informacji pogodowych poniżej mapy
            weather_icon = mpimg.imread(f"icons/{row['weather_icon']}")
            weather_img = OffsetImage(weather_icon, zoom=0.15)
            weather_ab = AnnotationBbox(weather_img, (0.1, 0.5), frameon=False, transform=desc_ax.transAxes)
            desc_ax.add_artist(weather_ab)
            annotation_boxes.append(weather_ab)

            temp_text = desc_ax.text(0.175, 0.5, f"{row['temp']}°C", ha='left', va='center', fontsize=22, weight='bold',
                                     transform=desc_ax.transAxes, fontfamily='monospace')
            weather_text = desc_ax.text(0.175, 0.05, f"{row['description']}", ha='center', va='center', fontsize=10,
                                        weight='normal',
                                        transform=desc_ax.transAxes, fontfamily='monospace')
            captions.append(temp_text)
            captions.append(weather_text)

            wind_icon = mpimg.imread(f"icons/wind_2.png")
            wind_icon = OffsetImage(wind_icon, zoom=0.15)
            wind_icon = AnnotationBbox(wind_icon, (0.45, 0.5), frameon=False, transform=desc_ax.transAxes)
            desc_ax.add_artist(wind_icon)
            annotation_boxes.append(wind_icon)
            wind_arrow = mpimg.imread(f"icons/{row['wind_arrow']}")
            wind_arrow = OffsetImage(wind_arrow, zoom=0.10)
            wind_arrow = AnnotationBbox(wind_arrow, (0.47, 0.20), frameon=False, transform=desc_ax.transAxes)
            desc_ax.add_artist(wind_arrow)
            annotation_boxes.append(wind_arrow)

            windspeed_text = desc_ax.text(0.525, 0.5, f"{row['windspeed']}km/h", ha='left', va='center', fontsize=22,
                                          weight='bold', transform=desc_ax.transAxes, fontfamily='monospace')
            captions.append(windspeed_text)

            if row['water']:
                if row['wave_arrow'] is not None and row['wave_height'] is not None:
                    if row['wave_height'] > 0:
                        wave_icon = mpimg.imread("icons/wave.png")
                        wave_icon = OffsetImage(wave_icon, zoom=0.15)
                        wave_icon = AnnotationBbox(wave_icon, (0.8, 0.5), frameon=False, transform=desc_ax.transAxes)
                        desc_ax.add_artist(wave_icon)
                        annotation_boxes.append(wave_icon)
                        wart_ = row['wave_height']
                        row['wave_height'] = round(wart_, 1)
                        wave_height_text = desc_ax.text(0.875, 0.5, f"{row['wave_height']}m", ha='left', va='center',
                                                        fontsize=22,
                                                        weight='bold', transform=desc_ax.transAxes,
                                                        fontfamily='monospace')
                        captions.append(wave_height_text)

            # Zapis klatki do pliku
            filename = f'frames/frame_{i}.png'
            plt.savefig(filename)
            filenames.append(filename)

        plt.close()

        clip = ImageSequenceClip(filenames, fps=1)  # Możesz dostosować fps (frames per second)
        clip.write_videofile("static/animation.mp4", codec="libx264")

        for filename in filenames:
            os.remove(filename)

    return render_template('page2.html', pogoda=df_new)


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    global file_content
    dane = []
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            with open(filepath, 'r') as f:
                plik = f.read().split()
                for row in plik:
                    dane.append(row.split(';'))
                data_converted = [[float(row[0]), float(row[1]), row[2], row[3]] for row in dane]
                # Get data from APIs
                dfs = []
                for row in data_converted:
                    time.sleep(0.25)
                    latitude_row, longitude_row, date_row, time_row = row
                    df1 = is_it_water(latitude_row, longitude_row, getkeys.IS_IT_WATER_KEY)
                    df2 = get_geocode_data(getkeys.GEO_CODING_KEY, latitude_row, longitude_row, date_row, time_row)
                    df3 = OpenMeteo(latitude_row, longitude_row, date_row, time_row)
                    df4 = VisualCrossing(latitude_row, longitude_row, date_row, time_row, getkeys.VISUAL_CROSSING_KEY)
                    combined_df = pd.concat([df1, df2, df3, df4], axis=1)
                    dfs.append(combined_df)

                file_content = pd.concat(dfs, ignore_index=True)
                file_content.index = pd.RangeIndex(start=0, stop=0 + len(file_content), step=1)
                file_content['index'] = pd.RangeIndex(start=0, stop=0 + len(file_content), step=1)
                cols = file_content.columns.tolist()
                cols = cols[-1:] + cols[:-1]
                file_content = file_content[cols]

                df_data = file_content.replace({np.nan: 'null'})

                data_list = df_data.to_dict(orient='records')

                for i, record in enumerate(data_list):
                    record = {'index': i, **record}
                    data_list[i] = record

                with open('dane.txt', 'w', encoding='utf-8') as f1:
                    f1.write(str(data_list))

            return redirect(url_for('index'))
    return render_template('upload.html', file_content=file_content)


@app.route('/wykresy')
def wykresy():
    global file_content
    global data

    # Konwersja danych do DataFrame
    if file_content is None:
        df = pd.DataFrame(data)
    else:
        df = file_content
    # Zamiana 'null' na NaN
    df.replace('null', np.nan, inplace=True)

    # Konwersja odpowiednich kolumn na typy liczbowe
    numeric_columns = ['latitude', 'longitude', 'wave_height', 'wave_direction', 'wave_period', 'temp', 'feelslike', 'humidity', 'precip', 'snow', 'pressure', 'windgust', 'windspeed', 'winddir', 'visibility', 'cloudcover', 'uvindex', 'moonphase']
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Konwersja kolumny 'date' na datetime
    df['date'] = pd.to_datetime(df['date'])

    # Ustawienie kolumny 'date' jako indeks
    df.set_index('date', inplace=True)

    # Usunięcie wierszy zawierających NaN w kluczowych kolumnach (tutaj: 'temp')
    df_clean = df.dropna(subset=['temp'])


    df.interpolate(method='linear', inplace=True)
    df['uvindex'] = pd.to_numeric(df['uvindex'], errors='coerce')
    df['temp'] = pd.to_numeric(df['temp'], errors='coerce')
    df['cloudcover'] = pd.to_numeric(df['cloudcover'], errors='coerce')
    df.dropna(subset=['temp', 'cloudcover'], inplace=True)

    # Tworzenie wykresu scatter plot
    fig = px.scatter(df, x='temp', y='cloudcover', size='uvindex', color='uvindex',
                     title='Wpływ temperatury i zachmurzenia na indeks UV',
                     labels={'temp': 'Temperatura (°C)', 'cloudcover': 'Zachmurzenie (%)'},
                     color_continuous_scale='Viridis',
                     hover_data=['uvindex'])

    # Dopasowanie marginesów dla osi x i y
    x_margin = (df['temp'].max() - df['temp'].min()) * 0.05
    y_margin = (df['cloudcover'].max() - df['cloudcover'].min()) * 0.05
    fig.update_xaxes(range=[df['temp'].min() - x_margin, df['temp'].max() + x_margin])
    fig.update_yaxes(range=[df['cloudcover'].min() - y_margin, df['cloudcover'].max() + y_margin])

    # Zapisz wykres do pliku HTML
    fig.write_html('static/uvindex_temperature_cloudcover.html', full_html=False)



    df['sunrise'] = pd.to_datetime(df['sunriseEpoch'], unit='s')
    df['sunset'] = pd.to_datetime(df['sunsetEpoch'], unit='s')

    # Obliczenie czasu trwania dnia i nocy
    df['daylight_duration'] = (df['sunset'] - df['sunrise']).dt.total_seconds() / 3600
    df['night_duration'] = 24 - df['daylight_duration']

    # Średnie czasu trwania dnia i nocy na dzień
    avg_daylight_duration = df.groupby('date')['daylight_duration'].mean().reset_index()
    avg_night_duration = df.groupby('date')['night_duration'].mean().reset_index()

    # Tworzenie wykresu
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=avg_daylight_duration['date'],
        y=avg_daylight_duration['daylight_duration'],
        name='Daylight Duration (hours)',
        marker_color='orange',
        showlegend=False
    ))

    fig.add_trace(go.Bar(
        x=avg_night_duration['date'],
        y=avg_night_duration['night_duration'],
        name='Night Duration (hours)',
        marker_color='black',
        showlegend=False
    ))

    fig.update_layout(
        title='Czas trwania dnia i nocy',
        xaxis_title='Data',
        yaxis_title='Czas trwania (h)',
        barmode='stack',
        hovermode='x',
    )

    # Zapisz wykres do pliku HTML
    fig.write_html('static/day_night_duration.html', full_html=False)



    df['precip'] = pd.to_numeric(df['precip'], errors='coerce')
    df['humidity'] = pd.to_numeric(df['humidity'], errors='coerce')

    # Określenie zakresów dla osi x i y na wykresie
    cloudcover_min = df['cloudcover'].min()
    cloudcover_max = df['cloudcover'].max()
    humidity_min = df['humidity'].min()
    humidity_max = df['humidity'].max()

    cloudcover_range = [cloudcover_min + 40, cloudcover_max + 5]
    humidity_range = [humidity_min + 30, humidity_max + 5]

    # Tworzenie wykresu za pomocą Plotly Express
    fig = px.scatter(df, x='cloudcover', y='humidity', size='precip', color='precip',
                     hover_data={'cloudcover': True, 'precip': True, 'humidity': True},
                     labels={'cloudcover': 'Pochmurność (%)', 'precip': 'Opady (mm)', 'humidity': 'Wilgotność (%)'},
                     title='Opady i wilgotność w zależności od pochmurności',
                     range_x=cloudcover_range,
                     range_y=humidity_range)

    fig.update_traces(marker=dict(line=dict(width=1, color='DarkSlateGrey')), selector=dict(mode='markers'))

    # Zapisz wykres do pliku HTML
    fig.write_html('static/cloudcover_humidity_precipitation.html', full_html=False)



    dates = [datetime.utcfromtimestamp(entry['datetimeEpoch']) for entry in data]
    feelslike = [float(entry['feelslike']) if entry['feelslike'] != 'null' else np.nan for entry in
                 data]  # Handle 'null' values
    cloudcover = [entry['cloudcover'] for entry in data]
    windspeed = [entry['windspeed'] for entry in data]

    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=cloudcover,
        y=windspeed,
        z=feelslike,
        mode='markers',
        marker=dict(
            size=5,
            color=feelslike,
            colorscale='Viridis',
            opacity=0.8,
            colorbar=dict(title='Temperatura odczuwalna (°C)')
        )
    ))

    fig.update_layout(
        title='Temperatura odczuwalna w zależności od zachmurzenia i prędkości wiatru',
        scene=dict(
            xaxis_title='Zachmurzenie (%)',
            yaxis_title='Prędkość wiatru (m/s)',
            zaxis_title='Temperatura odczuwalna (°C)'
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        height=600
    )

    # Zapisz wykres do pliku HTML
    fig.write_html('static/feelslike_cloudcover_windspeed.html', full_html=False)




    # Róża wiatrów dla kierunku wiatru
    def plot_wind_rose(direction, title, filename, cmap):
        # Usunięcie wierszy zawierających NaN
        direction = direction.dropna()

        # Konwersja na kąty w radianach
        theta = np.deg2rad(direction)

        # Utworzenie róży wiatrów
        ax = plt.subplot(111, polar=True)

        # Histogram
        n, bins, patches = ax.hist(theta, bins=16, edgecolor='black')

        # Kolory dla różnych segmentów histogramu
        for i, patch in enumerate(patches):
            patch.set_facecolor(cmap(i / len(patches)))

        # Oznaczenie kierunków
        ax.set_xticks(np.pi/180. * np.linspace(0, 360, 8, endpoint=False))
        ax.set_xticklabels(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'])

        ax.set_theta_direction(-1)
        ax.set_theta_offset(np.pi / 2.0)
        ax.set_title(title, va='bottom')

        # Zapisz wykres do pliku
        plt.savefig(f'static/{filename}')
        plt.close()

    # Kolory od żółtego do fioletowego
    cmap = plt.get_cmap('plasma')

    # Tworzenie róży wiatrów dla kierunku wiatru
    plot_wind_rose(df['winddir'], 'Róża wiatrów', 'wind_rose.png', cmap)

    # Tworzenie róży wiatrów dla kierunku fal
    plot_wind_rose(df['wave_direction'], 'Róża kierunku fal', 'wave_rose.png', cmap)

    # Wyodrębnienie godziny z kolumny 'time'
    df['hour'] = pd.to_datetime(df['time'], format='%H:%M').dt.hour

    # Usunięcie wierszy zawierających NaN w kolumnie 'visibility'
    df_clean = df.dropna(subset=['visibility'])

    # Grupowanie danych według godzin
    visibility_by_hour = df_clean.groupby('hour')['visibility'].mean().reset_index()

    # Dodanie kolumny z sformatowanymi godzinami
    visibility_by_hour['GODZINA'] = visibility_by_hour['hour'].apply(lambda x: f'{x:02d}:00')

    # Tworzenie wykresu kołowego za pomocą Plotly
    fig = px.pie(visibility_by_hour, values='visibility', names='GODZINA', title='Widoczność w zależności od godziny')
    fig.update_traces(textposition='inside')
    fig.update_layout(uniformtext_minsize=12, uniformtext_mode='hide')

    # Zapisz wykres do pliku z większym rozmiarem
    fig.write_html('static/visibility_pie.html', full_html=False)



    # Tworzenie wykresu punktowego wpływu prędkości wiatru na wysokość fal
    df_clean = df.dropna(subset=['wave_height', 'windspeed', 'winddir'])
    fig = px.scatter(df_clean,
                     x='windspeed', y='wave_height',
                     size='wave_height',  # Rozmiar punktów w zależności od wysokości fal
                     color='winddir',  # Kolor punktów według kierunku wiatru
                     hover_name='time',  # Tekst wyświetlany po najechaniu myszką
                     labels={
                         'windspeed': 'Prędkość Wiatru (m/s)',
                         'wave_height': 'Wysokość Fali (m)',
                         'winddir': 'Kierunek Wiatru (°)'
                     },
                     title='Wpływ Prędkości Wiatru na Wysokość Fal')
    fig.update_layout(coloraxis_colorbar=dict(title='Kierunek Wiatru (°)'))

    # Zapisz wykres do pliku HTML
    fig.write_html('static/wind_wave_scatter.html', full_html=False)



    # Usunięcie wierszy zawierających NaN w kolumnach kluczowych dla mapy temperatury
    df_clean = df.dropna(subset=['latitude', 'longitude', 'temp'])

    # Stworzenie mapy ciepła temperatury
    fig = px.density_mapbox(df_clean, lat='latitude', lon='longitude', z='temp', radius=10,
                            center=dict(lat=df_clean['latitude'].mean(), lon=df_clean['longitude'].mean()), zoom=5,
                            mapbox_style="open-street-map",
                            color_continuous_scale=px.colors.sequential.Jet,
                            # Skala kolorów od niebieskiego do czerwonego
                            title='Mapa ciepła temperatury')

    fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})

    # Zapisz wykres do pliku HTML
    fig.write_html('static/temperature_map.html', full_html=False)



    # Konwersja kolumn 'sunrise' i 'sunset' na datetime, ale zachowując tylko godziny
    df['sunrise_time'] = pd.to_datetime(df['sunrise'], format='%H:%M:%S').dt.time
    df['sunset_time'] = pd.to_datetime(df['sunset'], format='%H:%M:%S').dt.time

    # Przygotowanie danych do wizualizacji
    sunrise_df = df[['location', 'sunrise_time']].rename(columns={'sunrise_time': 'time'})
    sunset_df = df[['location', 'sunset_time']].rename(columns={'sunset_time': 'time'})

    # Dodanie kolumny 'event' aby rozróżnić wschód i zachód słońca
    sunrise_df['event'] = 'Wschód słońca'
    sunset_df['event'] = 'Zachód słońca'

    # Połączenie danych w jeden DataFrame
    plot_df = pd.concat([sunrise_df, sunset_df])

    # Stworzenie wykresu za pomocą Plotly
    fig = px.scatter(plot_df, x="location", y=plot_df['time'].astype(str), color="event",
                     title="Godziny wschodu i zachodu słońca w różnych krajach",
                     labels={"location": "Kraj", "time": "Godzina", "event": "Zdarzenie"},
                     hover_data={"location": True, "time": True},
                     color_discrete_map={"Wschód słońca": "orange", "Zachód słońca": "purple"})

    # Zapisz wykres do pliku HTML
    fig.write_html('static/sunrise_sunset.html', full_html=False)



    # Usunięcie wierszy z brakującymi wartościami
    df_clean = df.dropna(subset=['latitude', 'longitude', 'wave_period']).copy()

    # Konwersja kolumn na typ numeryczny
    df_clean.loc[:, 'latitude'] = pd.to_numeric(df_clean['latitude'], errors='coerce')
    df_clean.loc[:, 'longitude'] = pd.to_numeric(df_clean['longitude'], errors='coerce')
    df_clean.loc[:, 'wave_period'] = pd.to_numeric(df_clean['wave_period'], errors='coerce')

    # Ponowne usunięcie wierszy z brakującymi wartościami po konwersji
    df_clean = df_clean.dropna(subset=['latitude', 'longitude', 'wave_period'])

    # Stworzenie wykresu scatter_geo
    fig = px.scatter_geo(df_clean,
                         lat='latitude',
                         lon='longitude',
                         color='wave_period',
                         hover_name='location',
                         size='wave_period',
                         projection='natural earth',
                         title='Rozkład okresu fali w różnych lokalizacjach')

    # Zapisz wykres do pliku HTML
    fig.write_html('static/wave_period_map.html', full_html=False)



    # Usunięcie wierszy z brakującymi wartościami
    df_clean = df.dropna(subset=['windspeed', 'temp']).copy()

    # Stworzenie wykresu liniowego z gradientem
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_clean['windspeed'],
        y=df_clean['temp'],
        mode='lines+markers',
        line=dict(color='rosybrown', width=3),
        marker=dict(
            size=10,
            color=df_clean['temp'],  # gradient kolorów na podstawie temp
            colorscale='Portland',  # zmiana koloru odcieni
            showscale=True
        ),
    ))

    fig.update_layout(
        title='Temperatura w zależności od prędkości wiatru',
        xaxis_title='Prędkość wiatru (km/h)',
        yaxis_title='Temperatura (°C)',
        hovermode='closest',
        template='plotly_white'
    )

    # Zapisz wykres do pliku HTML
    fig.write_html('static/wind_speed_temperature.html', full_html=False)

    return render_template('wykresy.html')

def long_running_task(file_content):
    time.sleep(1)

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
