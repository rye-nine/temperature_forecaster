# to be used as API
import random

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

overriden_past_MAX_temps = {
            "LA": [2]
        }
overriden_past_MIN_temps = {
        "example": [42]
        }


from datetime import datetime, timedelta

#from meteostat import Point, Daily

from datetime import date, timedelta
import meteostat as ms

def get_temperatures(latitude: float, longitude: float, num_days: int, variable = "tmax"):
    end = date.today() - timedelta(days=1)  # Exclude today
    start = end - timedelta(days=num_days - 1)

    point = ms.Point(latitude, longitude)

    stations = ms.stations.nearby(point, limit=4)
    ts = ms.daily(stations, start, end)
    df = ms.interpolate(ts, point).fetch()

    #df[variable] = df[variable] * 9 / 5 + 32

    #print(f"Data: {list(df[variable])}")
    return [random.uniform(90,100) for i in range(num_days)]

