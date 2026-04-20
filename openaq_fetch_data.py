import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import requests
from dotenv import load_dotenv

# =========================
# Configuration
# =========================
load_dotenv()
API_KEY = os.getenv("OPENAQ_API_KEY", "")
BASE_URL = "https://api.openaq.org/v3"
OUTPUT_DIR = Path("final_data")


LOCATION_IDS: Dict[str, int] = {
    "The Views": 2981155,
    "Dubai Motor City": 3099020,
    "Serena": 3092518,
}


DESIRED_PARAMETERS = ["pm25", "pm10", "temperature", "humidity"]

# Fetching hourly data.
DATETIME_FROM = "2024-01-01T00:00:00Z"
DATETIME_TO = None  # None => current UTC time

# Query windows:
CHUNK_DAYS = 90
PAGE_LIMIT = 1000
MAX_RETRIES = 6
TIMEOUT = 60
SLEEP_BETWEEN_REQUESTS = 0.15

OUTPUT_PREFIX = "dubai_openaq"


# =========================
# Helpers
# =========================
def utc_now_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def parse_iso_datetime(value: str) -> datetime:
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    return datetime.fromisoformat(value)


def format_iso_datetime(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return (
        dt.astimezone(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def chunked_datetime_ranges(
    start_iso: str, end_iso: str, chunk_days: int
) -> Iterable[Tuple[str, str]]:
    start = parse_iso_datetime(start_iso)
    end = parse_iso_datetime(end_iso)
    current = start

    while current < end:
        nxt = min(current + timedelta(days=chunk_days), end)
        yield format_iso_datetime(current), format_iso_datetime(nxt)
        current = nxt


class OpenAQClient:
    def __init__(self, api_key: str, base_url: str = BASE_URL):
        if not api_key:
            raise ValueError("Missing OpenAQ API key")
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update(
            {
                "X-API-Key": api_key,
                "accept": "application/json",
                "user-agent": "dubai-openaq-downloader/1.0",
            }
        )

    def get_json(
        self, path: str, params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        url = f"{self.base_url}/{path.lstrip('/')}"

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = self.session.get(url, params=params, timeout=TIMEOUT)

                if resp.status_code == 429:
                    reset = resp.headers.get("x-ratelimit-reset")
                    wait_seconds = 5
                    if reset:
                        try:
                            wait_seconds = max(1, int(float(reset)))
                        except ValueError:
                            pass
                    time.sleep(wait_seconds)
                    continue

                if 500 <= resp.status_code < 600:
                    time.sleep(min(2**attempt, 30))
                    continue

                resp.raise_for_status()
                if SLEEP_BETWEEN_REQUESTS > 0:
                    time.sleep(SLEEP_BETWEEN_REQUESTS)
                return resp.json()

            except requests.RequestException:
                if attempt == MAX_RETRIES:
                    raise
                time.sleep(min(2**attempt, 30))

        raise RuntimeError("Unreachable retry loop")

    def paginate(
        self, path: str, params: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        params = dict(params or {})
        params.setdefault("limit", PAGE_LIMIT)
        page = 1
        results: List[Dict[str, Any]] = []

        while True:
            params["page"] = page
            payload = self.get_json(path, params=params)
            batch = payload.get("results", [])
            meta = payload.get("meta", {})

            if not batch:
                break

            results.extend(batch)

            found = meta.get("found")
            limit = meta.get("limit", params["limit"])
            try:
                found_int = int(found)
                limit_int = int(limit)
            except (TypeError, ValueError):
                found_int = None
                limit_int = params["limit"]

            if found_int is not None and page * limit_int >= found_int:
                break

            page += 1

        return results


@dataclass
class SensorChoice:
    sensor_id: int
    sensor_name: str
    parameter_name: str
    units: Optional[str]
    display_name: Optional[str]
    datetime_first_utc: Optional[str]
    datetime_last_utc: Optional[str]
    coverage_percent: Optional[float]


def get_location(client: OpenAQClient, location_id: int) -> Dict[str, Any]:
    payload = client.get_json(f"locations/{location_id}")
    results = payload.get("results", [])
    if not results:
        raise ValueError(f"No location found for location_id={location_id}")
    return results[0]


def get_sensor(client: OpenAQClient, sensor_id: int) -> Dict[str, Any]:
    payload = client.get_json(f"sensors/{sensor_id}")
    results = payload.get("results", [])
    if not results:
        raise ValueError(f"No sensor found for sensor_id={sensor_id}")
    return results[0]


def pick_best_sensors(
    client: OpenAQClient, location: Dict[str, Any], desired_parameters: Iterable[str]
) -> Dict[str, SensorChoice]:
    desired = set(desired_parameters)
    candidates: Dict[str, List[SensorChoice]] = {p: [] for p in desired}

    for sensor in location.get("sensors", []):
        parameter = sensor.get("parameter", {}) or {}
        name = str(parameter.get("name", "")).lower()
        if name not in desired:
            continue

        sensor_detail = get_sensor(client, int(sensor["id"]))
        coverage = sensor_detail.get("coverage", {}) or {}
        datetime_first = (sensor_detail.get("datetimeFirst") or {}).get("utc")
        datetime_last = (sensor_detail.get("datetimeLast") or {}).get("utc")

        candidates[name].append(
            SensorChoice(
                sensor_id=int(sensor["id"]),
                sensor_name=str(sensor.get("name", "")),
                parameter_name=name,
                units=parameter.get("units"),
                display_name=parameter.get("displayName"),
                datetime_first_utc=datetime_first,
                datetime_last_utc=datetime_last,
                coverage_percent=coverage.get("percentCoverage"),
            )
        )

    chosen: Dict[str, SensorChoice] = {}
    for parameter_name, options in candidates.items():
        if not options:
            continue

        def sort_key(item: SensorChoice) -> Tuple[int, str, str]:
            coverage = (
                item.coverage_percent if item.coverage_percent is not None else -1
            )
            dt_last = item.datetime_last_utc or ""
            dt_first = item.datetime_first_utc or ""
            return (coverage, dt_last, dt_first)

        chosen[parameter_name] = max(options, key=sort_key)

    return chosen


def trim_request_window(
    request_from: str, request_to: str, sensor: SensorChoice
) -> Optional[Tuple[str, str]]:
    start = parse_iso_datetime(request_from)
    end = parse_iso_datetime(request_to)

    if sensor.datetime_first_utc:
        start = max(start, parse_iso_datetime(sensor.datetime_first_utc))
    if sensor.datetime_last_utc:
        end = min(end, parse_iso_datetime(sensor.datetime_last_utc))

    if start >= end:
        return None

    return format_iso_datetime(start), format_iso_datetime(end)


def fetch_hourly_measurements(
    client: OpenAQClient,
    sensor: SensorChoice,
    datetime_from: str,
    datetime_to: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for chunk_from, chunk_to in chunked_datetime_ranges(
        datetime_from, datetime_to, CHUNK_DAYS
    ):
        params = {
            "datetime_from": chunk_from,
            "datetime_to": chunk_to,
            "limit": PAGE_LIMIT,
        }
        batch = client.paginate(
            f"sensors/{sensor.sensor_id}/measurements/hourly", params=params
        )
        rows.extend(batch)
        print(
            f"      fetched {len(batch):>6} rows for sensor {sensor.sensor_id} "
            f"({sensor.parameter_name}) from {chunk_from} to {chunk_to}"
        )

    return rows


def normalize_measurement_rows(
    rows: List[Dict[str, Any]],
    location: Dict[str, Any],
    sensor: SensorChoice,
) -> pd.DataFrame:
    normalized: List[Dict[str, Any]] = []

    for r in rows:
        period = r.get("period", {}) or {}
        dt_from = period.get("datetimeFrom", {}) or {}
        dt_to = period.get("datetimeTo", {}) or {}
        summary = r.get("summary", {}) or {}
        coverage = r.get("coverage", {}) or {}
        coords = r.get("coordinates", {}) or location.get("coordinates", {}) or {}
        parameter = r.get("parameter", {}) or {}
        flag_info = r.get("flagInfo", {}) or {}

        normalized.append(
            {
                "location_id": location.get("id"),
                "location_name": location.get("name"),
                "locality": location.get("locality"),
                "country_code": (location.get("country") or {}).get("code"),
                "timezone": location.get("timezone"),
                "provider": (location.get("provider") or {}).get("name"),
                "owner": (location.get("owner") or {}).get("name"),
                "latitude": coords.get("latitude"),
                "longitude": coords.get("longitude"),
                "sensor_id": sensor.sensor_id,
                "sensor_name": sensor.sensor_name,
                "parameter": parameter.get("name", sensor.parameter_name),
                "parameter_display_name": parameter.get(
                    "displayName", sensor.display_name
                ),
                "units": parameter.get("units", sensor.units),
                "value": r.get("value"),
                "period_label": period.get("label"),
                "period_interval": period.get("interval"),
                "datetime_utc_start": dt_from.get("utc"),
                "datetime_local_start": dt_from.get("local"),
                "datetime_utc_end": dt_to.get("utc"),
                "datetime_local_end": dt_to.get("local"),
                "summary_min": summary.get("min"),
                "summary_max": summary.get("max"),
                "summary_avg": summary.get("avg"),
                "summary_median": summary.get("median"),
                "coverage_expected_count": coverage.get("expectedCount"),
                "coverage_observed_count": coverage.get("observedCount"),
                "coverage_percent_complete": coverage.get("percentComplete"),
                "coverage_percent_coverage": coverage.get("percentCoverage"),
                "has_flags": flag_info.get("hasFlags"),
            }
        )

    df = pd.DataFrame(normalized)
    if not df.empty:
        df["datetime_utc_end"] = pd.to_datetime(
            df["datetime_utc_end"], utc=True, errors="coerce"
        )
        df["datetime_local_end"] = pd.to_datetime(
            df["datetime_local_end"], errors="coerce"
        )
    return df


def build_sensor_metadata_rows(
    location: Dict[str, Any], chosen_sensors: Dict[str, SensorChoice]
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for parameter_name, sensor in chosen_sensors.items():
        rows.append(
            {
                "location_id": location.get("id"),
                "location_name": location.get("name"),
                "parameter": parameter_name,
                "sensor_id": sensor.sensor_id,
                "sensor_name": sensor.sensor_name,
                "units": sensor.units,
                "display_name": sensor.display_name,
                "sensor_datetime_first_utc": sensor.datetime_first_utc,
                "sensor_datetime_last_utc": sensor.datetime_last_utc,
                "sensor_coverage_percent": sensor.coverage_percent,
            }
        )
    return rows


def make_daily_aggregate(hourly_df: pd.DataFrame) -> pd.DataFrame:
    if hourly_df.empty:
        return hourly_df.copy()

    df = hourly_df.copy()
    df["date_local"] = df["datetime_local_end"].dt.date

    group_cols = [
        "location_id",
        "location_name",
        "locality",
        "country_code",
        "timezone",
        "provider",
        "owner",
        "latitude",
        "longitude",
        "sensor_id",
        "sensor_name",
        "parameter",
        "parameter_display_name",
        "units",
        "date_local",
    ]

    daily = (
        df.groupby(group_cols, dropna=False)
        .agg(
            value_mean=("value", "mean"),
            value_median=("value", "median"),
            value_min=("value", "min"),
            value_max=("value", "max"),
            hourly_rows=("value", "size"),
            coverage_percent_complete_mean=("coverage_percent_complete", "mean"),
            coverage_percent_coverage_mean=("coverage_percent_coverage", "mean"),
        )
        .reset_index()
        .sort_values(["location_name", "parameter", "date_local"])
    )

    return daily


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    client = OpenAQClient(API_KEY)
    request_to = DATETIME_TO or utc_now_iso()

    all_hourly_frames: List[pd.DataFrame] = []
    sensor_meta_rows: List[Dict[str, Any]] = []
    location_rows: List[Dict[str, Any]] = []

    for label, location_id in LOCATION_IDS.items():
        print(f"\n=== {label} | location_id={location_id} ===")
        location = get_location(client, location_id)
        location_rows.append(
            {
                "requested_label": label,
                "location_id": location.get("id"),
                "location_name": location.get("name"),
                "locality": location.get("locality"),
                "country_code": (location.get("country") or {}).get("code"),
                "timezone": location.get("timezone"),
                "provider": (location.get("provider") or {}).get("name"),
                "owner": (location.get("owner") or {}).get("name"),
                "latitude": (location.get("coordinates") or {}).get("latitude"),
                "longitude": (location.get("coordinates") or {}).get("longitude"),
                "location_datetime_first_utc": (
                    location.get("datetimeFirst") or {}
                ).get("utc"),
                "location_datetime_last_utc": (location.get("datetimeLast") or {}).get(
                    "utc"
                ),
            }
        )

        print(f"Resolved location name: {location.get('name')}")
        print("Available sensors:")
        for s in location.get("sensors", []):
            p = s.get("parameter", {}) or {}
            print(
                f"  sensor_id={s.get('id'):<10} parameter={p.get('name')} units={p.get('units')}"
            )

        chosen_sensors = pick_best_sensors(client, location, DESIRED_PARAMETERS)
        sensor_meta_rows.extend(build_sensor_metadata_rows(location, chosen_sensors))

        if not chosen_sensors:
            print("  No desired parameters found at this location. Skipping.")
            continue

        print("Chosen sensors:")
        for parameter_name, sensor in chosen_sensors.items():
            print(
                f"  {parameter_name:<12} sensor_id={sensor.sensor_id:<10} "
                f"coverage={sensor.coverage_percent} first={sensor.datetime_first_utc} last={sensor.datetime_last_utc}"
            )

        for parameter_name, sensor in chosen_sensors.items():
            trimmed = trim_request_window(DATETIME_FROM, request_to, sensor)
            if trimmed is None:
                print(
                    f"  No overlap with requested time window for {parameter_name}. Skipping."
                )
                continue

            sensor_from, sensor_to = trimmed
            print(
                f"\n  Downloading HOURLY data for {parameter_name} from {sensor_from} to {sensor_to}"
            )
            rows = fetch_hourly_measurements(client, sensor, sensor_from, sensor_to)
            print(f"  Total hourly rows fetched: {len(rows)}")

            if rows:
                df = normalize_measurement_rows(rows, location, sensor)
                all_hourly_frames.append(df)

    if not all_hourly_frames:
        raise ValueError(
            "No data returned. Check API key, location IDs, parameters, or date range."
        )

    hourly_df = pd.concat(all_hourly_frames, ignore_index=True)
    hourly_df = hourly_df.sort_values(
        ["location_name", "parameter", "datetime_utc_end"]
    ).reset_index(drop=True)
    daily_df = make_daily_aggregate(hourly_df)

    sensors_df = pd.DataFrame(sensor_meta_rows).sort_values(
        ["location_name", "parameter"]
    )
    locations_df = pd.DataFrame(location_rows).sort_values(["location_name"])

    hourly_path = OUTPUT_DIR / f"{OUTPUT_PREFIX}_hourly.csv"
    daily_path = OUTPUT_DIR / f"{OUTPUT_PREFIX}_daily_from_hourly.csv"
    sensors_path = OUTPUT_DIR / f"{OUTPUT_PREFIX}_chosen_sensors.csv"
    locations_path = OUTPUT_DIR / f"{OUTPUT_PREFIX}_locations.csv"

    hourly_df.to_csv(hourly_path, index=False)
    daily_df.to_csv(daily_path, index=False)
    sensors_df.to_csv(sensors_path, index=False)
    locations_df.to_csv(locations_path, index=False)

    print("\nSaved files:")
    print(f"  {hourly_path}  ({len(hourly_df):,} rows)")
    print(f"  {daily_path}   ({len(daily_df):,} rows)")
    print(f"  {sensors_path} ({len(sensors_df):,} rows)")
    print(f"  {locations_path} ({len(locations_df):,} rows)")
    print(f"\nAll outputs are inside: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
