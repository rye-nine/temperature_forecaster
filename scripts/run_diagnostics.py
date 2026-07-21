from wipe_folder import wipe_folder
from temperature_forecaster.exploration import populate_charts
from temperature_forecaster.optimize_autoregression import optimize_autoregressive_terms 

view_charts = False # EDIT HOWEVER YOU WANT

if (wipe_folder("charts/diagnostics") & wipe_folder("charts/bias_variance")):
    populate_charts("tmax", view_charts)
    populate_charts("tmin", view_charts)
    print("now saving bias-variance charts")
    optimize_autoregressive_terms(variable="tmax", only_charts=True)
    optimize_autoregressive_terms(variable = "tmin", only_charts=True)
    print("Successfully executed run_diagnostics.py")
else:
    print("Unable to execute run_diagnostics.py: Invalid folder")