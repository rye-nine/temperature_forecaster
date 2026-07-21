from temperature_forecaster.__init__ import weather_station_coords
from temperature_forecaster.forecasting import run_forecasting
from datetime import datetime

empirical = False

# Returns an integer (e.g., 20)
day_number = datetime.now().day

for city_name in list(weather_station_coords.keys()):
    preset = 1 if not empirical else 2
    print(f"Here are the tmax probabilities for {city_name}:")
    t_max_temperature_lst = run_forecasting(preset,day_number + 1, minimum=80, maximum=110, city=city_name, variable="tmax")
    for tup in t_max_temperature_lst:
        print(f"{tup[0][0]} to {tup[0][1]-1}: {tup[1]}")
    print(f"Here are the tmin probabilities for {city_name}:")
    t_min_temperature_lst = run_forecasting(preset,day_number + 1, minimum=50, maximum=100, city=city_name, variable="tmax")
    for tup in t_min_temperature_lst:
        print(f"{tup[0][0]} to {tup[0][1]-1}: {tup[1]}")


