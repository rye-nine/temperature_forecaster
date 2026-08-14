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

from datetime import datetime, timedelta, date
import requests

# City definitions mapping to exact coordinates and primary ACIS station IDs (ICAO / GHCN)
CITY_CONFIG = {
    "LA": {"coords": (33.9425, -118.4081), "sid": "LAXthr"},
    "NYC": {"coords": (40.7789, -73.9692), "sid": "NYCthr"},
    "Chicago": {"coords": (41.7868, -87.7522), "sid": "ORDthr"},
    "Miami": {"coords": (25.7959, -80.2870), "sid": "MIAthr"},
    "Houston": {"coords": (29.6454, -95.2789), "sid": "IAHthr"},
    "Austin": {"coords": (30.1945, -97.6699), "sid": "ATTthr"},
    "Las Vegas": {"coords": (36.0801, -115.1522), "sid": "LASthr"},
    "Phoenix": {"coords": (33.4342, -112.0116), "sid": "PHXthr"},
}


def get_wrh_climate_temp(city: str, n: int, stat_type: str) -> list[float]:
    """Retrieves historical tmax or tmin data over the last 'n' days directly

    from the official NWS NOWData / ACIS climate station service.

    Parameters:
        city (str): Key matching CITY_CONFIG (e.g., "LA", "NYC").
        n (int): Number of past days to retrieve data for.
        stat_type (str): Either "tmax" or "tmin".

    Returns:
        list[float]: Daily temperatures in Â°F, ordered from most recent (index
        0) to oldest.
    """
    # 1. Input validation
    stat_type = stat_type.lower()
    if stat_type not in ["tmax", "tmin"]:
        raise ValueError("stat_type must be either 'tmax' or 'tmin'")
    if n <= 0:
        raise ValueError("n must be a positive integer")
    if city not in CITY_CONFIG:
        raise ValueError(
            f"City '{city}' not found. Choose from: {list(CITY_CONFIG.keys())}"
        )

    # 2. Derive date range (2 days ago back to 'n+1' days ago)
    today = datetime.now().date()
    end_date = today - timedelta(days=2)
    start_date = end_date - timedelta(days=n - 1)

    station_id = CITY_CONFIG[city]["sid"]

    # 3. Payload targeting ACIS StnData with explicit NWS Station Identifiers
    payload = {
        "sid": station_id,
        "sdate": start_date.strftime("%Y-%m-%d"),
        "edate": end_date.strftime("%Y-%m-%d"),
        "elems": [{"name": "maxt" if stat_type == "tmax" else "mint"}],
    }

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Content-Type": "application/json",
    }

    url = "https://data.rcc-acis.org/StnData"
    response = requests.post(url, json=payload, headers=headers, timeout=10)
    response.raise_for_status()

    data = response.json()

    if "data" not in data or not data["data"]:
        raise ValueError(
            f"No climate data returned for {city} from ACIS service."
        )

    # 4. Extract temperatures from returned daily list [["YYYY-MM-DD", "value"], ...]
    temperatures = []
    for entry in data["data"]:
        val = entry[1]
        try:
            temperatures.append(float(val))
        except (ValueError, TypeError):
            # Skip missing data flags (e.g., "M" for missing or "T" for trace)
            print("Skipping date due to missing value")
            continue

    # 5. Reverse list so index 0 is most recent and last is oldest
    temperatures.reverse()

    return temperatures

    
month_map = {
    "JAN": 1,
    "FEB": 2,
    "MAR": 3,
    "APR": 4,
    "MAY": 5,
    "JUN": 6,
    "JUL": 7,
    "AUG": 8,
    "SEP": 9,
    "OCT": 10,
    "NOV": 11,
    "DEC": 12
}

def get_day(market_ticker: str) -> int:
    split_ticker = market_ticker.split("-")
    str_date = split_ticker[1]
    year = 2000 + int(str_date[:2])
    month = month_map[str_date[2:5]]
    day = int(str_date[-2:])
    datetime_object = date(year, month, day)
    return datetime_object.timetuple().tm_yday  



