from datetime import timedelta
from pathlib import Path

import meteostat as ms
import pandas as pd

# ==========================================
# CONFIG
# ==========================================
DATA_DIR = Path("final_data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

OPENAQ_HOURLY_PATH = DATA_DIR / "dubai_openaq_hourly.csv"
OPENAQ_LOCATIONS_PATH = DATA_DIR / "dubai_openaq_locations.csv"
OUTPUT_METEO_HOURLY_PATH = DATA_DIR / "meteostat_hourly_all_locations.csv"
OUTPUT_METEO_DAILY_PATH = DATA_DIR / "meteostat_daily_all_locations.csv"
OUTPUT_MASTER_HOURLY_PATH = DATA_DIR / "master_hourly_from_openaq_meteostat.csv"
OUTPUT_MASTER_DAILY_PATH = DATA_DIR / "master_daily_from_openaq_meteostat.csv"

# OpenAQ parameters to keep from our 3-location file
OPENAQ_PARAMETERS = ["pm25", "pm10", "temperature"]


# ==========================================
# HELPERS
# ==========================================
def normalize_local_hour(series: pd.Series) -> pd.Series:
    s = pd.to_datetime(series, errors="coerce")
    if getattr(s.dt, "tz", None) is not None:
        s = s.dt.tz_localize(None)
    return s.dt.floor("h")


def to_naive_local_hour(series: pd.Series) -> pd.Series:
    """
    This will Parse datetimes, convert timezone-aware timestamps to naive local time,
    and floor to the hour. This makes OpenAQ and Meteostat easier to merge.
    """
    dt = pd.to_datetime(series, errors="coerce")
    try:
        # Works when timestamps are timezone-aware
        return dt.dt.tz_localize(None).dt.floor("h")
    except TypeError:
        # Works when timestamps are already naive
        return dt.dt.floor("h")


def load_locations_from_openaq(hourly: pd.DataFrame) -> pd.DataFrame:
    cols = ["location_name", "latitude", "longitude", "timezone"]
    if OPENAQ_LOCATIONS_PATH.exists():
        locs = pd.read_csv(OPENAQ_LOCATIONS_PATH)
        rename_map = {}
        if "location_name" not in locs.columns and "name" in locs.columns:
            rename_map["name"] = "location_name"
        locs = locs.rename(columns=rename_map)
        available = [c for c in cols if c in locs.columns]
        if len(available) == len(cols):
            return (
                locs[cols]
                .drop_duplicates()
                .sort_values("location_name")
                .reset_index(drop=True)
            )

    return (
        hourly[cols]
        .drop_duplicates()
        .sort_values("location_name")
        .reset_index(drop=True)
    )


def build_openaq_hourly_wide(hourly: pd.DataFrame) -> pd.DataFrame:
    required = {
        "location_name",
        "latitude",
        "longitude",
        "timezone",
        "parameter",
        "value",
        "datetime_local_end",
    }
    missing = required - set(hourly.columns)
    if missing:
        raise ValueError(f"Missing required columns in OpenAQ hourly file: {missing}")

    df = hourly.copy()
    df = df[df["parameter"].isin(OPENAQ_PARAMETERS)].copy()
    df["datetime_local"] = normalize_local_hour(df["datetime_local_end"])
    df["date"] = pd.to_datetime(df["datetime_local"]).dt.date

    # Long -> wide
    wide = df.pivot_table(
        index=[
            "location_name",
            "datetime_local",
            "date",
            "latitude",
            "longitude",
            "timezone",
        ],
        columns="parameter",
        values="value",
        aggfunc="mean",
    ).reset_index()
    wide.columns.name = None

    #  hourly quality summaries from OpenAQ
    if "coverage_percent_complete" in df.columns:
        coverage = df.groupby(
            ["location_name", "datetime_local", "parameter"], as_index=False
        ).agg(coverage_percent_complete=("coverage_percent_complete", "mean"))
        coverage_wide = coverage.pivot_table(
            index=["location_name", "datetime_local"],
            columns="parameter",
            values="coverage_percent_complete",
            aggfunc="mean",
        ).reset_index()
        coverage_wide.columns.name = None
        coverage_wide = coverage_wide.rename(
            columns={
                "pm25": "pm25_coverage_percent_complete",
                "pm10": "pm10_coverage_percent_complete",
                "temperature": "temperature_coverage_percent_complete",
            }
        )
        wide = wide.merge(
            coverage_wide, on=["location_name", "datetime_local"], how="left"
        )

    return wide.sort_values(["location_name", "datetime_local"]).reset_index(drop=True)


def fetch_meteostat_hourly_for_locations(
    locations: pd.DataFrame, start_dt, end_dt
) -> pd.DataFrame:
    all_hourly = []

    for _, row in locations.iterrows():
        location_name = row["location_name"]
        lat = float(row["latitude"])
        lon = float(row["longitude"])
        tz = row["timezone"] if pd.notna(row["timezone"]) else "Asia/Dubai"

        print(f"Fetching Meteostat hourly for {location_name} ({lat}, {lon}) [{tz}]")

        point = ms.Point(lat, lon)

        # Find nearby stations first
        stations = ms.stations.nearby(point, limit=4)

        # Get hourly data from those stations
        ts = ms.hourly(stations, start_dt, end_dt, timezone=tz)

        # Interpolate station data to the point
        ts_interp = ms.interpolate(ts, point)

        weather = ts_interp.fetch()

        # Defensive guard in case the local install still returns None
        if weather is None or weather.empty:
            print(f"  No Meteostat hourly data returned for {location_name}")
            continue

        weather = weather.reset_index().rename(columns={"time": "datetime_local"})
        weather["datetime_local"] = normalize_local_hour(weather["datetime_local"])
        weather["date"] = pd.to_datetime(weather["datetime_local"]).dt.date
        weather["location_name"] = location_name

        weather = weather.rename(
            columns={
                "temp": "temp_meteostat_hourly",
                "rhum": "humidity_meteostat_hourly",
                "prcp": "precipitation_meteostat_hourly",
                "wspd": "wind_speed_meteostat_hourly",
                "pres": "pressure_meteostat_hourly",
                "cldc": "cloud_cover_meteostat_hourly",
            }
        )

        keep_cols = [
            "location_name",
            "datetime_local",
            "date",
            "temp_meteostat_hourly",
            "humidity_meteostat_hourly",
            "precipitation_meteostat_hourly",
            "wind_speed_meteostat_hourly",
            "pressure_meteostat_hourly",
            "cloud_cover_meteostat_hourly",
        ]
        existing = [c for c in keep_cols if c in weather.columns]
        all_hourly.append(weather[existing].copy())

    if not all_hourly:
        raise ValueError("No Meteostat hourly data was fetched for any location.")

    return (
        pd.concat(all_hourly, ignore_index=True)
        .sort_values(["location_name", "datetime_local"])
        .reset_index(drop=True)
    )


def aggregate_meteostat_daily(meteostat_hourly: pd.DataFrame) -> pd.DataFrame:
    daily = (
        meteostat_hourly.groupby(["location_name", "date"], as_index=False)
        .agg(
            temp_meteostat=("temp_meteostat_hourly", "mean"),
            temp_min_meteostat=("temp_meteostat_hourly", "min"),
            temp_max_meteostat=("temp_meteostat_hourly", "max"),
            humidity_meteostat=("humidity_meteostat_hourly", "mean"),
            precipitation_meteostat=("precipitation_meteostat_hourly", "sum"),
            wind_speed_meteostat=("wind_speed_meteostat_hourly", "mean"),
            pressure_meteostat=("pressure_meteostat_hourly", "mean"),
            cloud_cover_meteostat=("cloud_cover_meteostat_hourly", "mean"),
        )
        .sort_values(["location_name", "date"])
        .reset_index(drop=True)
    )
    return daily


def build_master_hourly(
    openaq_hourly_wide: pd.DataFrame, meteostat_hourly: pd.DataFrame
) -> pd.DataFrame:
    openaq_hourly_wide = openaq_hourly_wide.copy()
    meteostat_hourly = meteostat_hourly.copy()

    openaq_hourly_wide["datetime_local"] = normalize_local_hour(
        openaq_hourly_wide["datetime_local"]
    )
    meteostat_hourly["datetime_local"] = normalize_local_hour(
        meteostat_hourly["datetime_local"]
    )

    openaq_hourly_wide["date"] = pd.to_datetime(openaq_hourly_wide["date"]).dt.date
    meteostat_hourly["date"] = pd.to_datetime(meteostat_hourly["date"]).dt.date

    print("OpenAQ datetime dtype:", openaq_hourly_wide["datetime_local"].dtype)
    print("Meteostat datetime dtype:", meteostat_hourly["datetime_local"].dtype)

    master_hourly = openaq_hourly_wide.merge(
        meteostat_hourly,
        on=["location_name", "datetime_local", "date"],
        how="left",
    )

    master_hourly["humidity"] = master_hourly.get("humidity_meteostat_hourly")

    desired_order = [
        "location_name",
        "datetime_local",
        "date",
        "latitude",
        "longitude",
        "timezone",
        "humidity",
        "pm10",
        "pm25",
        "temperature",
        "temp_meteostat_hourly",
        "humidity_meteostat_hourly",
        "precipitation_meteostat_hourly",
        "wind_speed_meteostat_hourly",
        "pressure_meteostat_hourly",
        "cloud_cover_meteostat_hourly",
        "pm25_coverage_percent_complete",
        "pm10_coverage_percent_complete",
        "temperature_coverage_percent_complete",
    ]
    existing = [c for c in desired_order if c in master_hourly.columns]
    others = [c for c in master_hourly.columns if c not in existing]
    master_hourly = master_hourly[existing + others]

    return master_hourly.sort_values(["location_name", "datetime_local"]).reset_index(
        drop=True
    )


def build_master_daily(master_hourly: pd.DataFrame) -> pd.DataFrame:
    daily = (
        master_hourly.groupby(
            ["location_name", "date", "latitude", "longitude", "timezone"],
            as_index=False,
        )
        .agg(
            humidity=("humidity", "mean"),
            pm10=("pm10", "mean"),
            pm25=("pm25", "mean"),
            temperature=("temperature", "mean"),
            temp_meteostat=("temp_meteostat_hourly", "mean"),
            temp_min_meteostat=("temp_meteostat_hourly", "min"),
            temp_max_meteostat=("temp_meteostat_hourly", "max"),
            humidity_meteostat=("humidity_meteostat_hourly", "mean"),
            precipitation_meteostat=("precipitation_meteostat_hourly", "sum"),
            wind_speed_meteostat=("wind_speed_meteostat_hourly", "mean"),
            pressure_meteostat=("pressure_meteostat_hourly", "mean"),
            cloud_cover_meteostat=("cloud_cover_meteostat_hourly", "mean"),
            hourly_rows=("datetime_local", "size"),
            pm25_non_null_hours=("pm25", lambda s: s.notna().sum()),
            pm10_non_null_hours=("pm10", lambda s: s.notna().sum()),
            temperature_non_null_hours=("temperature", lambda s: s.notna().sum()),
        )
        .sort_values(["location_name", "date"])
        .reset_index(drop=True)
    )

    desired_order = [
        "location_name",
        "date",
        "latitude",
        "longitude",
        "timezone",
        "humidity",
        "pm10",
        "pm25",
        "temperature",
        "temp_meteostat",
        "temp_min_meteostat",
        "temp_max_meteostat",
        "humidity_meteostat",
        "precipitation_meteostat",
        "wind_speed_meteostat",
        "pressure_meteostat",
        "cloud_cover_meteostat",
        "hourly_rows",
        "pm25_non_null_hours",
        "pm10_non_null_hours",
        "temperature_non_null_hours",
    ]
    existing = [c for c in desired_order if c in daily.columns]
    others = [c for c in daily.columns if c not in existing]
    daily = daily[existing + others]

    return daily


def main() -> None:
    if not OPENAQ_HOURLY_PATH.exists():
        raise FileNotFoundError(
            f"OpenAQ hourly file not found: {OPENAQ_HOURLY_PATH}\n"
            " we need to have final_data/dubai_openaq_hourly.csv"
        )

    hourly = pd.read_csv(OPENAQ_HOURLY_PATH)
    locations = load_locations_from_openaq(hourly)

    print("Locations found:")
    print(locations)

    openaq_hourly_wide = build_openaq_hourly_wide(hourly)

    start_dt = (
        pd.to_datetime(openaq_hourly_wide["datetime_local"]).min().to_pydatetime()
    )
    end_dt = pd.to_datetime(openaq_hourly_wide["datetime_local"]).max().to_pydatetime()

    # Add a tiny buffer on the end so the final day is captured reliably
    end_dt = end_dt + timedelta(hours=1)

    print(f"\nOpenAQ local datetime range: {start_dt} to {end_dt}")

    meteostat_hourly = fetch_meteostat_hourly_for_locations(locations, start_dt, end_dt)
    meteostat_daily = aggregate_meteostat_daily(meteostat_hourly)

    master_hourly = build_master_hourly(openaq_hourly_wide, meteostat_hourly)
    master_daily = build_master_daily(master_hourly)

    meteostat_hourly.to_csv(OUTPUT_METEO_HOURLY_PATH, index=False)
    meteostat_daily.to_csv(OUTPUT_METEO_DAILY_PATH, index=False)
    master_hourly.to_csv(OUTPUT_MASTER_HOURLY_PATH, index=False)
    master_daily.to_csv(OUTPUT_MASTER_DAILY_PATH, index=False)

    print("\nSaved files:")
    print(f"  {OUTPUT_METEO_HOURLY_PATH} ({len(meteostat_hourly):,} rows)")
    print(f"  {OUTPUT_METEO_DAILY_PATH} ({len(meteostat_daily):,} rows)")
    print(f"  {OUTPUT_MASTER_HOURLY_PATH} ({len(master_hourly):,} rows)")
    print(f"  {OUTPUT_MASTER_DAILY_PATH} ({len(master_daily):,} rows)")

    print("\nMaster hourly columns:")
    print(master_hourly.columns.tolist())

    print("\nMaster daily columns:")
    print(master_daily.columns.tolist())

    print("\nMaster daily head:")
    print(master_daily.head())


if __name__ == "__main__":
    main()
