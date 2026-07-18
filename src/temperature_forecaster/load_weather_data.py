"""
load_weather_data.py

Author: Rye
Date: 2026-06-30

Description:
    Loads historical weather data for the temperature
    forecasting project. Only includes the "tmax" and "tmin" columns, 
    which represent the daily maximum and minimum temperatures, respectively.
    Also removes any rows with missing values in these columns.

Responsibilities:
    - Import raw weather data from CSV files
    - Remove rows with missing temperature values
    - Export to data/raw/ for further analysis

Functions:
    load_data()
        Loads weather data for all configured locations.

    create_fourier_features(loaded_data)
        Adds Fourier terms using the optimal number of harmonics
        for each weather station.

Dependencies:
    - pandas
    - numpy

Notes:
    Assumes that all raw data files are stored in:

        data/raw/

    and that each DataFrame uses a DatetimeIndex.
""" 

from datetime import date
import meteostat as ms # type: ignore
from temperature_forecaster.__init__ import weather_station_coords
from temperature_forecaster.find_optimize_fourier_terms import optimize_fourier_terms
from temperature_forecaster.paths import DATA_RAW


def getCoords(locationName):
    return weather_station_coords[locationName]

def celsius_to_fahrenheit(celsius):
    return (celsius * 9/5) + 32

def load_df(location):

    today = date.today()
    coords = getCoords(location)
    POINT = ms.Point(coords[0], coords[1])

    # ok please change this potentially
    START = date(today.year-10, today.month, today.day)
    END = date(today.year, today.month, today.day)
    stations = ms.stations.nearby(POINT, limit=1)

    ts = ms.daily(stations, START, END)
    df = ms.interpolate(ts, POINT).fetch()
    
    df["tmin"] = df["tmin"].interpolate(method = "time")
    df["tmax"] = df["tmax"].interpolate(method = "time")

    df[["tmax", "tmin"]] = celsius_to_fahrenheit(df[["tmax", "tmin"]])
    
    return df

def load_data():
    for location in weather_station_coords.keys():
        df = load_df(location)
        df.to_csv(DATA_RAW / f"{location}_weather_data.csv")
        print(f"Stored in data/raw: {location}_weather_data.csv")
