import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.gridspec import GridSpec
from moviepy.editor import ImageSequenceClip


def getAnimation(df):
    os.makedirs('frames', exist_ok=True)

    # Obliczanie zakresów geograficznych
    min_lat = df['latitude'].min()
    max_lat = df['latitude'].max()
    min_lon = df['longitude'].min()
    max_lon = df['longitude'].max()

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

    for i, row in df.iterrows():
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
        ab = AnnotationBbox(imagebox, (row['longitude'], row['latitude']), frameon=False, transform=ccrs.PlateCarree())
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
        weather_text = desc_ax.text(0.175, 0.05, f"{row['description']}", ha='center', va='center', fontsize=10, weight='normal',
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

                    wave_height_text = desc_ax.text(0.875, 0.5, f"{row['wave_height']}m", ha='left', va='center',
                                                    fontsize=22,
                                                    weight='bold', transform=desc_ax.transAxes, fontfamily='monospace')
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