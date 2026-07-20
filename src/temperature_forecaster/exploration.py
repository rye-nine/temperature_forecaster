# im a little concerned that i copy and pasted a lot (like the city index stuff)

import altair as alt
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import os
from pathlib import Path
import io 
import base64
import pickle

from temperature_forecaster.probability_model import get_final_residuals
from temperature_forecaster.__init__ import weather_station_coords
from temperature_forecaster.paths import PROJECT_ROOT

def get_ts(city, df_list, variable = "tmax"):
    city_index = (list(weather_station_coords.keys())).index(city)
    city_df = df_list[city_index]

    set_width = 1500
    set_height = 800
    
    #actual
    c1 = alt.Chart(city_df).mark_line().encode(
        x = "time",
        y = variable 
            ).properties(
                    width = set_width,
                    height = set_height
                    )
    # estimation
    c2 = alt.Chart(city_df).mark_line(color="red").encode(
            x = "time",
            y = "final_prediction:Q" # tmin predictions not up yet
            ).properties(
                    width = set_width,
                    height = set_height
                    )

    # residuals
    c3 = alt.Chart(city_df).mark_line(color="green").encode(
            x = "time",
            y = "final_residuals:Q"
            ).properties(
                    width = set_width,
                    height = set_height
                    )
    return_df = c1+c2+c3
    return c1+c2+c3

def get_histogram(city, df_list, variable = "tmax"):
    city_index = (list(weather_station_coords.keys())).index(city)
    city_df = df_list[city_index]

    histogram = alt.Chart(city_df).mark_bar().encode(
        alt.X("final_residuals:Q", bin=alt.Bin(maxbins=50), title="Residual (°F)"),
        alt.Y("count()", title="Frequency")
        )
    return histogram

def get_Q_Q_plot(city, df_list, variable = "tmax"):
    city_index = (list(weather_station_coords.keys())).index(city)
    city_df = df_list[city_index]

    fig, ax = plt.subplots()

    stats.probplot(city_df["final_residuals"], dist="norm", plot=ax)

    return fig

def get_charts(city = None, variable = "tmax"):

    list_of_df = get_final_residuals(variable)

    list_df = [dataf.reset_index() for dataf in list_of_df] 
    print(list_df[2].columns)

    if (city is None):
        city_names = list(weather_station_coords.keys())
        lst = [[get_ts(nameCity, list_df, variable),
                        get_histogram(nameCity, list_df, variable),
                        get_Q_Q_plot(nameCity, list_df, variable)] for nameCity in city_names]
        
        return_dict = {
            cityName : cityCharts
            for cityName, cityCharts in zip(city_names, lst)
                }
        return return_dict

    ts_charts = get_ts(city, list_df, variable)
    histogram = get_histogram(city, list_df, variable)
    QQ_plot = get_Q_Q_plot(city, list_df, variable)

    buffer = io.BytesIO()
    QQ_plot.savefig(buffer, format="png", bbox_inches="tight")
    buffer.seek(0)
    qqplot_base64 = base64.b64encode(buffer.read()).decode("utf-8")

    combined = ts_charts & histogram
    combined_html = combined.to_html()
    #ts_html = ts_charts.to_html()
    #hist_html = histogram.to_html()

    html = f"""
    <html>

    <body>

    <h1>{city}_{variable} Forecast Diagnostics</h1>

    {combined_html}

    <hr>

    <h2>QQ Plot</h2>

    <img src="data:image/png;base64,{qqplot_base64}">

    </body>

    </html>
    """

    #ts_charts.save(f"charts/{city}_.html")
    #os.startfile(f"{city}.html")  # Windows only
    print(f"Created charts for {city}")

    with open(PROJECT_ROOT / f"charts/diagnostics/{city}_{variable}.html", "w", encoding="utf-8") as f:
        f.write(html)

    return html

def populate_charts(variable = "tmax", open_charts = False): # used for populating the charts/ folder
    city_names = list(weather_station_coords.keys())
    for city in city_names:
        get_charts(city, variable)
        if (open_charts):
            output = PROJECT_ROOT / f"charts/diagnostics/{city}_{variable}.html" # formerly Path("charts") / ...
            os.startfile(output) # Windows only
    print("Open a chart by running the following command in root folder: start charts/[city_name].html")
