#from temperature_forecaster.evaluation import form_groups
from temperature_forecaster.forecasting import run_forecasting
import pandas as pd 

my_list = [1,2,3,4,5]
my_series = pd.Series(my_list)
print(my_series)
print(type(my_series))

print(run_forecasting(1, my_series, 90,92, "Miami", "tmax"))

#form_groups(1, 90, 92, "Miami", "tmax")  # on kalshi this is market "90 to 91"
