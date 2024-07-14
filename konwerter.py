import csv
from datetime import datetime

# Słownik do mapowania polskich nazw miesięcy na angielskie
polish_to_english_month = {
    "sty": "Jan", "lut": "Feb", "mar": "Mar", "kwi": "Apr",
    "maj": "May", "cze": "Jun", "lip": "Jul", "sie": "Aug",
    "wrz": "Sep", "paź": "Oct", "lis": "Nov", "gru": "Dec"
}

# Funkcja do przekształcania formatu daty i czasu
def convert_datetime(datetime_str):
    # Zamiana polskich nazw miesięcy na angielskie
    for polish, english in polish_to_english_month.items():
        datetime_str = datetime_str.replace(polish, english)
    # Zdefiniowanie formatu wejściowego
    input_format = "%d %b %Y%H:%M:%S"
    # Przekształcenie na obiekt datetime
    dt = datetime.strptime(datetime_str, input_format)
    # Zdefiniowanie formatu wyjściowego
    output_date_format = "%Y-%m-%d"
    output_time_format = "%H:%M"
    return dt.strftime(output_date_format), dt.strftime(output_time_format)

# Otwieranie pliku CSV i przekształcanie danych
with open('wspolrzedne.csv', 'r', newline='') as csvfile:
    csvreader = csv.reader(csvfile, delimiter='\t')
    
    # Otwieranie pliku TXT do zapisu
    with open('wspolrzedne.txt', 'w', newline='') as txtfile:
        for row in csvreader:
            date_time_str, lat, lon = row
            date_str, time_str = convert_datetime(date_time_str)
            # Zapisywanie w formacie: lat;lon;date;time
            txtfile.write(f"{lat};{lon};{date_str};{time_str}\n")