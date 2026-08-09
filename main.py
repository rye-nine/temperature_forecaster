"""
THIS WAS FORKED FROM JOSH KIM'S INFRASTRUCTURE WITH PERMISSION
This is a sample strategy.py file. Edit it to fit your strategy!

In order to run your tests, open the terminal in this directory and run "python3 -i strategy.py"
Once the interactive session opens, type in the specific tests you would like to run. 
    Ensure all tests succeed locally before making a pull request.

These tests serve as basic sanity checks for your strategy. 
    They are unable to detect if the outputs returned are "correct" or not.
    Thus, it is YOUR job to read through their outputs and ensure they match your intended behavior.
"""
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

from temperature_forecaster.__init__ import weather_station_coords, get_day
from scripts.wipe_folder import wipe_folder

# ✅
# series = [s.upper() for s in ["kxhighlax", "kxhighny", "kxhighchi", "kxhighmia",
#           "kxhighthou", "kxhighaus", "kxhightlv", "kxhightphx",
#           "kxlowtlax", "kxlowtnyc", "kxlowtchi", "kxlowtmia",
#           "kxlowthou", "kxlowtaus", "kxlowtlv", "kxlowtphx"]]  # eg. ["kxhighlax", "kxhighchi"]

series = [s.upper() for s in ["kxhighchi"]]

## A list of the series tickers you will be trading on.

funds_per_market = {
    s: 1 for s in series
}
## How many $ to use in each bet for each series.

min_edge = {
    s: 0.09 for s in series
}
## Calculate each market's price to buy at.
## You may update this dynamically in pipeline_live or use static constants.

sell_at_distance = {
    s: 0.03 for s in series
}
## The cent difference from model prediction that you'd like to sell at.

use_market_data = False
## Whether to use live 1000ms market data or not.

def collect_live(series_ticker):
    """
    Collects and returns all data necessary to make a live prediction for {series_ticker}.

    @param series_ticker: The ticker of the series you would like to collect data for.

    @return: Data 
    @type return: Any

    > collect_live("kxhighlax")
    > Data (what this looks like will depend on your pipeline)
    """
    ticker_to_city_dict = dict.fromkeys(weather_station_coords)
    ticker_to_city_dict["LA"] = ["kxhighlax", "kxlowtlax"]
    ticker_to_city_dict["NYC"] = ["kxhighny", "kxlowtnyc"]
    ticker_to_city_dict["Chicago"] = ["kxhighchi", "kxlowtchi"]
    ticker_to_city_dict["Miami"] = ["kxhighmia", "kxlowtmia"]
    ticker_to_city_dict["Houston"] = ["kxhighthou", "kxlowthou"]
    ticker_to_city_dict["Austin"] = ["kxhighaus", "kxlowtaus"]
    ticker_to_city_dict["Las Vegas"] = ["kxhightlv", "kxlowtlv"]
    ticker_to_city_dict["Phoenix"] = ["kxhightphx", "kxlowtphx"]

    for city_name in ticker_to_city_dict.keys():
        ticker_to_city_dict[city_name] = [s.upper() for s in ticker_to_city_dict[city_name]]

    tickers_to_city = {tuple(value): key for key, value in ticker_to_city_dict.items()}


    target_city = None 
    for tup in tickers_to_city.keys():
        if series_ticker.upper() in tup:
            target_city = tickers_to_city[tup]
    if (target_city is None): # if we can't find the ticker
        print("City not found!")
        raise ValueError

    variable = "tmax" if ("high" in series_ticker or "HIGH" in series_ticker) else "tmin"

    from temperature_forecaster.load_weather_data import load_data
    load_data(target_city)
    from temperature_forecaster.fourier_features import engineer_and_store_data
    engineer_and_store_data(variable, target_city)

    wipe_folder("models/fourier_models")
    from temperature_forecaster.fourier_training import train_and_store_models
    train_and_store_models(variable, target_city)

    wipe_folder("models/residual_autoregression_models")
    from temperature_forecaster.residual_autocorrelation import train_and_store_autocorrelations
    train_and_store_autocorrelations(variable, target_city)

    #current_dir = os.path.dirname(os.path.abspath(__file__))    
    #subprocess.run([sys.executable, "-m", "src.regenerate_data.py"], cwd=current_dir)
    #subprocess.run([sys.executable, "-m", "src.recompute_fourier_models.py"], cwd=current_dir)
    #subprocess.run([sys.executable, "-m", "src.recompute_AR_models.py"], cwd=current_dir)
    #subprocess.run([sys.executable, "-m", "src.run_diagnostics.py"], cwd=current_dir)

    return target_city, variable

def pipeline_live(series_ticker, market_tickers):
    """
    Uses collected live data to run an end-to-end prediction pipeline. Returns the {series_ticker}'s predictions as a dict.

    @param series_ticker: The ticker of the series you would like to run your pipeline for.
    @param market_tickers: The market tickers of this series.
    
    @return: A dictionary with market tickers as keys and predictions as values.
    @type return: dict[str, double]

    > pipeline_live("kxhighlax", ["market_1", "market_2", "market_3"])
    > {"market_1": 0.9, "market_2": 0.2, ..., "market_9": 0.67}
    """

    MODE = 2 # 1 for normal distribution approximation, 2 for empirical residual distribution
    # use normal distribution in fall and spring
    # use empirical residual distribution in summer and winter because there may be extreme values

    #today = datetime.now(ZoneInfo("America/Los_Angeles")).date()
    #day_of_year = today.timetuple().tm_yday

    day_of_year = get_day(market_tickers[0])
    #from src.forecasting import appropriate_intervals, sliced_intervals
    #print(sliced_intervals(market_tickers))
    #return
    city_target, target_variable = collect_live(series_ticker)
    # get models
    from temperature_forecaster.forecasting import kalshi_forecasting, run_forecasting, appropriate_intervals, sliced_intervals

    print(f"City: {city_target}")
    #return_thing = kalshi_forecasting(1,day_of_year, market_tickers, city=city_target, variable=target_variable)
    return_thing = kalshi_forecasting(MODE,day_of_year, market_tickers, city=city_target, variable=target_variable)

    from temperature_forecaster.backtesting import backtest

    backtest("2024-01-01", city = city_target, variable = target_variable)
    # wipe_folder("charts/diagnostics") 
    # wipe_folder("charts/bias_variance")

    from temperature_forecaster.evaluation import calibrate

    calibrate(77,79, city_name = city_target, variable_name = target_variable)
    
    
    from temperature_forecaster.exploration import populate_charts
    populate_charts(target_variable, open_charts=True, city=city_target)

    return return_thing

def heartbeat(market_ticker, data): # optional, only if you're using kalshi data
    """
    Every 1000ms, this is called with the current market_ticker's data.
    Use this function to collect data / etc.

    @param market_ticker: The ticker of the market being tracked.
    @param data: The corresponding data. print() for info.
    """
    raise NotImplementedError

##### ALTERNATIVELY, run python3 strategy.py to run all tests at once.
from datetime import datetime, timezone

def collect_live_test(series_ticker="KXHIGHMIA"):
    print(f"\n===== COLLECT_LIVE TEST =====")
    result = collect_live(series_ticker)
    assert result is not None
    print(result)
    return result

def pipeline_live_test(series_ticker="kxhighlax", market_tickers=["KXHIGHLAX-26AUG04-T75", 
  "KXHIGHLAX-26AUG04-B75.5",
  "KXHIGHLAX-26AUG04-B77.5", 
  "KXHIGHLAX-26AUG04-B79.5", 
  "KXHIGHLAX-26AUG04-B81.5",
  "KXHIGHLAX-26AUG04-T82"]):
    print(f"\n===== PIPELINE_LIVE TEST =====")
    result = pipeline_live(series_ticker, market_tickers)
    assert isinstance(result, dict)
    print(result)
    return result

def heartbeat_test(market_ticker="KXBTC15M-26JUL190400-00", data=None):
    print(f"\n===== HEARTBEAT TEST =====")
    data = data if data is not None else {}
    heartbeat(market_ticker, data)

if __name__ == "__main__":
    import sys
    import traceback

    tests = [
        #("collect_live", collect_live_test),
        ("pipeline_live", pipeline_live_test)
        #("heartbeat", heartbeat_test),
    ]

    failures = []
    for name, test_fn in tests:
        try:
            test_fn()
        except Exception:
            failures.append(name)
            print(f"\n===== {name.upper()} FAILED =====")
            traceback.print_exc()

    print("\n===== SUMMARY =====")
    if failures:
        print(f"FAILED: {', '.join(failures)}")
        sys.exit(1)
    else:
        print("All tests passed.")
        sys.exit(0)
