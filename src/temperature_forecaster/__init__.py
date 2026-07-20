# to be used as API

weather_station_coords = {
    "LA": (33.9425, -118.4081),
    "NYC": (40.7789, -73.9692),
    "Chicago": (41.7868, -87.7522),
    "Miami": (25.7959, -80.2870),
    "Houston": (29.6454, -95.2789),
    "Austin": (30.1945, -97.6699),
    "Las Vegas": (36.0801, -115.1522),
    "Phoenix": (33.4342, -112.0116)
}

from datetime import datetime, timedelta

#from meteostat import Point, Daily

def get_temperatures(latitude: float, longitude: float, days: int, variable = "tmax"):
    """
    Returns the daily max/min temperatures for the past `days` days.

    Temperatures are returned in Fahrenheit.
    """
    weather_station_coords = {
        "LA": (33.9425, -118.4081),
        "NYC": (40.7789, -73.9692),
        "Chicago": (41.7868, -87.7522),
        "Miami": (25.7959, -80.2870),
        "Houston": (29.6454, -95.2789),
        "Austin": (30.1945, -97.6699),
        "Las Vegas": (36.0801, -115.1522),
        "Phoenix": (33.4342, -112.0116)
    }

from datetime import date, timedelta
import meteostat as ms

def get_temperatures(latitude: float, longitude: float, num_days: int, variable = "tmax"):
    """
    Returns the daily max/min temperatures for the past `days` days.

    Temperatures are returned in Fahrenheit.
    """
    point = ms.Point(latitude, longitude)

    end = date.today()
    start = end - timedelta(days=num_days)

    stations = ms.stations.nearby(point, limit=4)

    ts = ms.daily(stations, start, end)
    df = ms.interpolate(ts, point).fetch()

    df[variable] = df[variable] * 9 / 5 + 32

    print(list(df[variable]))
    return list(df[variable])

