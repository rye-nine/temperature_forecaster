# im a little concerned that i copy and pasted a lot (like the city index stuff)

import altair as alt
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

from src.temperature_forecaster.probability_model import get_final_residuals
from src.temperature_forecaster.__init__ import weather_station_coords

def get_ts(city, df_list, variable = "tmax"):
    city_index = (list(weather_station_coords.keys())).index(city)
    city_df = df_list[city_index]
    
    if (variable == "tmin"):
        raise NotImplementedError
    #actual
    c1 = alt.Chart(city_df).mark_line().encode(
        x = "time",
        y = variable 
            ).properties(
                    width = 1200
                    )
    # estimation
    c2 = alt.Chart(city_df).mark_line(color="red").encode(
            x = "time",
            y = "final_prediction:Q" # tmin predictions not up yet
            ).properties(
                    width = 1200
                    )

    # residuals
    c3 = alt.Chart(city_df).mark_line(color="green").encode(
            x = "time",
            y = "final_residuals:Q"
            ).properties(
                    width = 1200
                    )
    return_df = c1+c2+c3
    return return_df

def get_histogram(city, df_list, variable = "tmax"):
    city_index = (list(weather_station_coords.keys())).index(city)
    city_df = df_list[city_index]

    if (variable == "tmin"):
        raise NotImplementedError

    histogram = alt.Chart(city_df).mark_bar().encode(
        alt.X("final_residuals:Q", bin=alt.Bin(maxbins=50), title="Residual (°F)"),
        alt.Y("count()", title="Frequency")
        )
    return histogram

def get_Q_Q_plot(city, df_list, variable = "tmax"):
    city_index = (list(weather_station_coords.keys())).index(city)
    city_df = df_list[city_index]

    if (variable == "tmin"):
        raise NotImplementedError

    fig, ax = plt.subplots()

    stats.probplot(city_df["residual"], dist="norm", plot=ax)

    return fig

def get_charts(city = None, variable = "tmax"):

    list_df = get_final_residuals("tmax")

    if (city is None):
        city_names = list(weather_station_coords.keys())
        return_list = [[get_ts(nameCity, list_df, variable),
                        get_histogram(nameCity, list_df, variable),
                        get_Q_Q_plot(nameCity, list_df, variable)] for nameCity in city_names]
        return return_list

    ts_charts = get_ts(city, list_df, variable)
    histogram = get_histogram(city, list_df, variable)
    QQ_plot = get_Q_Q_plot(city, list_df, variable)

    return ts_charts, histogram, QQ_plot
