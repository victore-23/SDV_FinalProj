import requests
import pandas as pd
from typing import Dict, List, Any

# Define the OpenAQ API URL and headers
API_KEY = ""  # Replace with your actual API key from OpenAQ
BASE_URL = "https://api.openaq.org/v3"
HEADERS = {
    "X-API-Key": API_KEY,
    "accept": "application/json",
}

# Parameter names we are interested in
DESIRED_PARAMETERS = {
    "pm25": ["pm25"],
    "pm10": ["pm10"],
    "temperature": ["temperature", "temp"],
    "humidity": ["humidity", "rh", "relativehumidity"],
}

# Function to make API requests
def get_json(url: str, params: Dict[str, Any] | None = None) -> Dict[str, Any]:
    resp = requests.get(url, headers=HEADERS, params=params, timeout=60)
    resp.raise_for_status()
    return resp.json()

# Function to fetch location details
def get_location(location_id: int) -> Dict[str, Any]:
    data = get_json(f"{BASE_URL}/locations/{location_id}")
    results = data.get("results", [])
    if not results:
        raise ValueError(f"No location found for location_id={location_id}")
    return results[0]

# Function to select relevant sensors based on the parameters we want
def pick_sensors(location: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    sensors = location.get("sensors", [])
    chosen: Dict[str, Dict[str, Any]] = {}

    for sensor in sensors:
        parameter_obj = sensor.get("parameter", {})
        param_name = str(parameter_obj.get("name", "")).lower()

        for canonical_name, aliases in DESIRED_PARAMETERS.items():
            if canonical_name in chosen:
                continue
            if param_name in aliases:
                chosen[canonical_name] = {
                    "sensor_id": sensor["id"],
                    "sensor_name": sensor.get("name"),
                    "parameter_name": param_name,
                    "units": parameter_obj.get("units"),
                    "display_name": parameter_obj.get("displayName"),
                }

    return chosen

# Function to fetch daily data for a specific sensor
def fetch_sensor_days(sensor_id: int, date_from: str, date_to: str) -> List[Dict[str, Any]]:
    all_rows: List[Dict[str, Any]] = []
    page = 1
    limit = 1000

    while True:
        params = {
            "date_from": date_from,
            "date_to": date_to,
            "limit": limit,
            "page": page,
        }
        data = get_json(f"{BASE_URL}/sensors/{sensor_id}/days", params=params)
        results = data.get("results", [])
        meta = data.get("meta", {})

        if not results:
            break

        all_rows.extend(results)

        found = meta.get("found", 0)
        current_limit = meta.get("limit", limit)

        try:
            found_num = int(found)
        except (TypeError, ValueError):
            found_num = None

        if found_num is not None and page * current_limit >= found_num:
            break

        page += 1

    return all_rows

# Function to normalize the fetched data into a DataFrame
def normalize_day_rows(
    rows: List[Dict[str, Any]],
    canonical_parameter: str,
    sensor_info: Dict[str, Any],
    location: Dict[str, Any],
) -> pd.DataFrame:
    normalized = []

    for r in rows:
        period = r.get("period", {}) or {}
        dt_from = period.get("datetimeFrom", {}) or {}
        dt_to = period.get("datetimeTo", {}) or {}
        coverage = r.get("coverage", {}) or {}
        summary = r.get("summary", {}) or {}

        normalized.append(
            {
                "location_id": location.get("id"),
                "location_name": location.get("name"),
                "timezone": location.get("timezone"),
                "country_code": (location.get("country") or {}).get("code"),
                "sensor_id": sensor_info["sensor_id"],
                "sensor_name": sensor_info["sensor_name"],
                "parameter": canonical_parameter,
                "parameter_source_name": sensor_info["parameter_name"],
                "units": sensor_info["units"],
                "period_label": period.get("label"),
                "date_local_start": dt_from.get("local"),
                "date_local_end": dt_to.get("local"),
                "date_utc_start": dt_from.get("utc"),
                "date_utc_end": dt_to.get("utc"),
                "value": r.get("value"),
                "summary_min": summary.get("min"),
                "summary_max": summary.get("max"),
                "summary_median": summary.get("median"),
                "coverage_expected_count": coverage.get("expectedCount"),
                "coverage_observed_count": coverage.get("observedCount"),
                "coverage_percent_complete": coverage.get("percentComplete"),
                "coverage_percent_coverage": coverage.get("percentCoverage"),
            }
        )

    return pd.DataFrame(normalized)

# Main function to download and process the data
def main() -> None:
    location_ids = {
        "Dubai Motor City": 2981155,
        "The Views": 3099020,
        "Serena": 3092518
    }

    date_from = "2019-01-01"  # Start date for historical data
    date_to = "2026-04-18"    # End date (today)

    all_frames = []  # To collect data frames for each sensor

    for location_name, location_id in location_ids.items():
        print(f"Fetching location details for {location_name}...")
        location = get_location(location_id)

        print(f"Location: {location.get('name')} (ID: {location.get('id')})")
        print("Available sensors:")
        for s in location.get("sensors", []):
            p = s.get("parameter", {})
            print(f"  sensor_id={s.get('id'):<10} parameter={p.get('name')} units={p.get('units')}")

        chosen = pick_sensors(location)

        if not chosen:
            raise ValueError(f"No matching sensors found for {location_name}.")

        print(f"\nChosen sensors for {location_name}:")
        for k, v in chosen.items():
            print(f"  {k}: sensor_id={v['sensor_id']} ({v['sensor_name']}, source_param={v['parameter_name']})")

        frames = []
        for canonical_parameter, sensor_info in chosen.items():
            print(f"\nDownloading daily data for {canonical_parameter}...")
            rows = fetch_sensor_days(sensor_info["sensor_id"], date_from, date_to)
            print(f"  rows fetched: {len(rows)}")

            if rows:
                df_part = normalize_day_rows(rows, canonical_parameter, sensor_info, location)
                frames.append(df_part)

        if frames:
            df_long = pd.concat(frames, ignore_index=True)
            all_frames.append(df_long)

    if not all_frames:
        raise ValueError("No data was returned from any of the selected sensors/date ranges.")

    # Combine all data frames into one
    final_df = pd.concat(all_frames, ignore_index=True)
    final_df.to_csv("dubai_openaq_combined_data.csv", index=False)
    print("\nSaved: dubai_openaq_combined_data.csv")

    print("\nDone.")

if __name__ == "__main__":
    main()