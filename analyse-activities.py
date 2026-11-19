import argparse
import gzip
import hashlib
import io
import json
import os
import time
import zipfile
from datetime import date, datetime, timedelta
from pathlib import Path

import fitdecode
import numpy as np
import pandas as pd
import requests

from src import config as app_config
from src import db as app_db

# --- COLD-START PHYSIOLOGY & PATH SETTINGS ---
DEFAULT_FTP = app_config.DEFAULT_FTP        # Cold-start Functional Threshold Power in Watts
DEFAULT_MAX_HR = app_config.DEFAULT_MAX_HR  # Cold-start Maximum Heart Rate in bpm
STRAVA_DIR = app_config.STRAVA_DIR
OUTPUT_CSV = app_config.CSV_PATH
TOKEN_FILE = app_config.TOKEN_FILE

# Strava API Credentials (can be set via environment variables or inside .strava_tokens.json)
STRAVA_CLIENT_ID = app_config.STRAVA_CLIENT_ID
STRAVA_CLIENT_SECRET = app_config.STRAVA_CLIENT_SECRET
STRAVA_REFRESH_TOKEN = app_config.STRAVA_REFRESH_TOKEN

# Intervals.icu API Credentials
INTERVALS_API_KEY = app_config.INTERVALS_API_KEY
INTERVALS_ATHLETE_ID = app_config.INTERVALS_ATHLETE_ID

BASE_COLS = [
    "Activity ID", "Activity Date", "Activity Name", "Activity Type",
    "Activity Description", "Elapsed Time", "Moving Time", "Distance",
    "Elevation Gain", "Average Heart Rate", "Max Heart Rate",
    "Average Watts", "Weighted Average Power", "Relative Effort"
]

METRIC_COLS = [
    "fit_lap_count",
    "effective_ftp", "effective_max_hr",
    "hr_z1_pct", "hr_z2_pct", "hr_z3_pct", "hr_z4_pct", "hr_z5_pct",
    "avg_power_fit", "normalized_power", "intensity_factor", "variability_index",
    "pwr_z1_pct", "pwr_z2_pct", "pwr_z3_pct", "pwr_z4_pct", "pwr_z5_plus_pct",
    "peak_1m_w", "peak_5m_w", "peak_20m_w", "hiit_surges_count",
    "aerobic_decoupling_pct", "climbing_time_pct"
]


# =====================================================================
# 1. CORE METRIC ENGINE (Shared by FIT Parser and Strava API Streams)
# =====================================================================
def extract_stream_peaks(df: pd.DataFrame) -> dict:
    """
    Extracts raw power peaks (1m, 5m, 20m) and a spike-filtered 5-second rolling
    mean clean_max_hr (bounded between 135 and 208 bpm) from a 1Hz stream DataFrame.
    """
    peaks = {
        "peak_1m_w": np.nan,
        "peak_5m_w": np.nan,
        "peak_20m_w": np.nan,
        "clean_max_hr": np.nan,
    }
    if df.empty:
        return peaks

    pwr = pd.to_numeric(df["power"], errors="coerce").dropna()
    if len(pwr) > 0:
        for label, secs in [("peak_1m_w", 60), ("peak_5m_w", 300), ("peak_20m_w", 1200)]:
            if len(pwr) >= secs:
                peaks[label] = round(float(pwr.rolling(window=secs).mean().max()), 1)

    hr = pd.to_numeric(df["hr"], errors="coerce").dropna()
    if len(hr) >= 5:
        hr_5s = hr.rolling(window=5, min_periods=5).mean().dropna()
        hr_5s_bounded = hr_5s[(hr_5s >= 135) & (hr_5s <= 208)]
        if not hr_5s_bounded.empty:
            peaks["clean_max_hr"] = round(float(hr_5s_bounded.max()), 1)

    return peaks


def _valid_num(val) -> float | None:
    if val is None:
        return None
    if isinstance(val, str):
        val = val.strip().replace(",", "")
        if not val:
            return None
    try:
        f = float(val)
        return None if np.isnan(f) else f
    except (TypeError, ValueError):
        return None


def resolve_dynamic_physiology(
    activity_ts: pd.Timestamp | None,
    history_records: list[dict],
    current_peaks: dict | None = None,
    last_known_ftp: int | None = None,
) -> tuple[int, int]:
    """
    Resolves (effective_ftp, effective_max_hr) using trailing rolling windows:
      - effective_ftp: round(max(0.95 * peak_20m_w, 0.80 * peak_5m_w)) over trailing 90 days
        up to activity_ts (inclusive of current_peaks), carrying forward last_known_ftp
        if no power peaks exist in the 90-day window, then falling back to DEFAULT_FTP.
      - effective_max_hr: max of valid HR (135 <= hr <= 208) over trailing 365 days
        up to activity_ts (inclusive of current_peaks), falling back to DEFAULT_MAX_HR.
    """
    current_peaks = current_peaks or {}

    if activity_ts is None or pd.isna(activity_ts):
        ftp_candidates = []
        p20 = _valid_num(current_peaks.get("peak_20m_w"))
        p5 = _valid_num(current_peaks.get("peak_5m_w"))
        if p20 and p20 > 0:
            ftp_candidates.append(0.95 * p20)
        if p5 and p5 > 0:
            ftp_candidates.append(0.80 * p5)
        eff_ftp = int(round(max(ftp_candidates))) if ftp_candidates else (last_known_ftp or DEFAULT_FTP)

        hr_candidates = [
            h for h in (_valid_num(current_peaks.get("clean_max_hr")), _valid_num(current_peaks.get("max_hr")))
            if h is not None and 135 <= h <= 208
        ]
        eff_max_hr = int(round(max(hr_candidates))) if hr_candidates else DEFAULT_MAX_HR
        return eff_ftp, eff_max_hr

    ftp_cutoff = activity_ts - pd.Timedelta(days=90)
    hr_cutoff = activity_ts - pd.Timedelta(days=365)

    ftp_candidates: list[float] = []
    hr_candidates: list[float] = []
    latest_hist_ftp: int | None = last_known_ftp

    for rec in history_records:
        rec_ts = rec.get("ts")
        if rec_ts is None or pd.isna(rec_ts) or rec_ts > activity_ts:
            continue

        rec_ftp = _valid_num(rec.get("effective_ftp"))
        if rec_ftp and rec_ftp > 0:
            latest_hist_ftp = int(round(rec_ftp))

        if rec_ts >= ftp_cutoff:
            p20 = _valid_num(rec.get("peak_20m_w"))
            p5 = _valid_num(rec.get("peak_5m_w"))
            if p20 and p20 > 0:
                ftp_candidates.append(0.95 * p20)
            if p5 and p5 > 0:
                ftp_candidates.append(0.80 * p5)

        if rec_ts >= hr_cutoff:
            for key in ("clean_max_hr", "max_hr", "Max Heart Rate"):
                h = _valid_num(rec.get(key))
                if h is not None and 135 <= h <= 208:
                    hr_candidates.append(h)

    p20_cur = _valid_num(current_peaks.get("peak_20m_w"))
    p5_cur = _valid_num(current_peaks.get("peak_5m_w"))
    if p20_cur and p20_cur > 0:
        ftp_candidates.append(0.95 * p20_cur)
    if p5_cur and p5_cur > 0:
        ftp_candidates.append(0.80 * p5_cur)

    for key in ("clean_max_hr", "max_hr", "Max Heart Rate"):
        h_cur = _valid_num(current_peaks.get(key))
        if h_cur is not None and 135 <= h_cur <= 208:
            hr_candidates.append(h_cur)

    if ftp_candidates:
        eff_ftp = int(round(max(ftp_candidates)))
    elif latest_hist_ftp and latest_hist_ftp > 0:
        eff_ftp = int(latest_hist_ftp)
    else:
        eff_ftp = DEFAULT_FTP

    eff_max_hr = int(round(max(hr_candidates))) if hr_candidates else DEFAULT_MAX_HR
    return eff_ftp, eff_max_hr


def compute_stream_metrics(
    df: pd.DataFrame,
    lap_count: int = 0,
    effective_ftp: int | None = None,
    effective_max_hr: int | None = None,
    peaks: dict | None = None,
) -> dict:
    """
    Computes training metrics from a 1Hz DataFrame containing:
    ['hr', 'power', 'cadence', 'speed', 'altitude']
    """
    ftp = int(effective_ftp) if effective_ftp and effective_ftp > 0 else DEFAULT_FTP
    max_hr = int(effective_max_hr) if effective_max_hr and effective_max_hr > 0 else DEFAULT_MAX_HR

    metrics = {col: np.nan for col in METRIC_COLS}
    metrics["fit_lap_count"] = lap_count
    metrics["effective_ftp"] = ftp
    metrics["effective_max_hr"] = max_hr

    if df.empty:
        return metrics

    extracted_peaks = peaks if peaks is not None else extract_stream_peaks(df)
    for label in ("peak_1m_w", "peak_5m_w", "peak_20m_w"):
        metrics[label] = extracted_peaks.get(label, np.nan)

    # 1. Heart Rate Zones & Distribution (% of recorded samples)
    hr = pd.to_numeric(df["hr"], errors="coerce").dropna()
    if len(hr) > 60:
        hr_pct = hr / max_hr
        metrics["hr_z1_pct"] = round((hr_pct < 0.60).mean() * 100, 1)
        metrics["hr_z2_pct"] = round(((hr_pct >= 0.60) & (hr_pct < 0.70)).mean() * 100, 1)
        metrics["hr_z3_pct"] = round(((hr_pct >= 0.70) & (hr_pct < 0.80)).mean() * 100, 1)
        metrics["hr_z4_pct"] = round(((hr_pct >= 0.80) & (hr_pct < 0.90)).mean() * 100, 1)
        metrics["hr_z5_pct"] = round((hr_pct >= 0.90).mean() * 100, 1)

    # 2. Power Metrics: NP, IF, VI, Zones & HIIT Surges
    pwr = pd.to_numeric(df["power"], errors="coerce").dropna()
    if len(pwr) > 60:
        avg_pwr = pwr.mean()
        rolling_30s = pwr.rolling(window=30, min_periods=1).mean()
        np_watts = (rolling_30s.pow(4).mean()) ** 0.25

        metrics["avg_power_fit"] = round(avg_pwr, 1)
        metrics["normalized_power"] = round(np_watts, 1)
        metrics["intensity_factor"] = round(np_watts / ftp, 2)
        metrics["variability_index"] = round(np_watts / avg_pwr, 2) if avg_pwr > 0 else np.nan

        # Power Zones (% of time)
        pwr_pct = pwr / ftp
        metrics["pwr_z1_pct"] = round((pwr_pct < 0.55).mean() * 100, 1)
        metrics["pwr_z2_pct"] = round(((pwr_pct >= 0.55) & (pwr_pct < 0.75)).mean() * 100, 1)
        metrics["pwr_z3_pct"] = round(((pwr_pct >= 0.75) & (pwr_pct < 0.90)).mean() * 100, 1)
        metrics["pwr_z4_pct"] = round(((pwr_pct >= 0.90) & (pwr_pct < 1.05)).mean() * 100, 1)
        metrics["pwr_z5_plus_pct"] = round((pwr_pct >= 1.05).mean() * 100, 1)

        # HIIT / Surge Counter (Distinct efforts > 120% FTP lasting at least 15 seconds)
        above_120 = (pwr > (ftp * 1.20)).astype(int)
        blocks = (above_120.diff() == 1).cumsum()
        surge_durations = above_120.groupby(blocks).sum()
        metrics["hiit_surges_count"] = int((surge_durations >= 15).sum())

    # 3. Aerobic Decoupling (First Half vs. Second Half Power-to-HR Ratio)
    both = df[["power", "hr"]].apply(pd.to_numeric, errors="coerce").dropna()
    both = both[(both["power"] > 30) & (both["hr"] > 60)]
    if len(both) >= 1200:  # Only meaningful on sessions >= 20 mins
        mid = len(both) // 2
        first_half, second_half = both.iloc[:mid], both.iloc[mid:]
        ef_1 = first_half["power"].mean() / first_half["hr"].mean()
        ef_2 = second_half["power"].mean() / second_half["hr"].mean()
        if ef_1 > 0:
            metrics["aerobic_decoupling_pct"] = round(((ef_1 - ef_2) / ef_1) * 100, 2)

    # 4. Climbing Time (% of moving time spent on gradients > 3%)
    alt = pd.to_numeric(df["altitude"], errors="coerce").dropna()
    spd = pd.to_numeric(df["speed"], errors="coerce").dropna()
    if len(alt) > 60 and len(spd) > 60:
        alt_diff = alt.rolling(10, min_periods=1).mean().diff()
        dist_step = spd.reindex(alt.index).fillna(0)
        valid = dist_step > 1.5  # moving > 5.4 km/h
        if valid.sum() > 30:
            grade = (alt_diff[valid] / dist_step[valid]) * 100
            metrics["climbing_time_pct"] = round((grade > 3.0).mean() * 100, 1)

    return metrics


# =====================================================================
# 2. SOURCE ADAPTER A: LOCAL .FIT / .FIT.GZ FILES & RAW FIT BYTES
# =====================================================================
def _parse_fit_stream(stream, source_name: str = "stream") -> tuple[pd.DataFrame, int]:
    """Extracts second-by-second records and lap count from a binary FIT file-like stream."""
    records = []
    lap_count = 0

    try:
        with fitdecode.FitReader(stream) as fit:
            for frame in fit:
                if isinstance(frame, fitdecode.FitDataMessage):
                    if frame.name == "lap":
                        lap_count += 1
                    elif frame.name == "record":
                        records.append({
                            "timestamp": frame.get_value("timestamp", fallback=None),
                            "hr": frame.get_value("heart_rate", fallback=np.nan),
                            "power": frame.get_value("power", fallback=np.nan),
                            "cadence": frame.get_value("cadence", fallback=np.nan),
                            "speed": frame.get_value(
                                "enhanced_speed",
                                fallback=frame.get_value("speed", fallback=np.nan)
                            ),
                            "altitude": frame.get_value(
                                "enhanced_altitude",
                                fallback=frame.get_value("altitude", fallback=np.nan)
                            ),
                        })
    except Exception as e:
        print(f"Warning: Could not parse {source_name}: {e}")
        return pd.DataFrame(), 0

    if not records:
        return pd.DataFrame(), lap_count

    return pd.DataFrame(records), lap_count


def parse_fit_bytes(raw_bytes: bytes, source_name: str = "bytes") -> tuple[pd.DataFrame, int]:
    """Extracts second-by-second records and lap count from raw .fit, .fit.gz, or .zip bytes."""
    if not raw_bytes:
        return pd.DataFrame(), 0

    if raw_bytes[:4] == b"PK\x03\x04":
        try:
            with zipfile.ZipFile(io.BytesIO(raw_bytes)) as zf:
                fit_names = [n for n in zf.namelist() if n.lower().endswith(".fit")]
                if not fit_names:
                    print(f"Warning: No .fit file found inside ZIP archive for {source_name}")
                    return pd.DataFrame(), 0
                with zf.open(fit_names[0]) as fit_member:
                    return _parse_fit_stream(fit_member, source_name=fit_names[0])
        except Exception as e:
            print(f"Warning: Could not unpack ZIP for {source_name}: {e}")
            return pd.DataFrame(), 0

    if raw_bytes[:2] == b"\x1f\x8b":
        try:
            with gzip.GzipFile(fileobj=io.BytesIO(raw_bytes)) as gz:
                return _parse_fit_stream(gz, source_name=source_name)
        except Exception as e:
            print(f"Warning: Could not gunzip {source_name}: {e}")
            return pd.DataFrame(), 0

    return _parse_fit_stream(io.BytesIO(raw_bytes), source_name=source_name)


def parse_fit_records(file_path: Path) -> tuple[pd.DataFrame, int]:
    """Extracts second-by-second records and lap count from a .fit or .fit.gz file."""
    if not file_path.exists():
        return pd.DataFrame(), 0

    opener = gzip.open if file_path.suffix == ".gz" else open
    try:
        with opener(file_path, "rb") as f:
            return _parse_fit_stream(f, source_name=file_path.name)
    except Exception as e:
        print(f"Warning: Could not open {file_path.name}: {e}")
        return pd.DataFrame(), 0


def parse_fit_metrics(
    file_path: Path,
    effective_ftp: int | None = None,
    effective_max_hr: int | None = None,
) -> dict:
    """Extracts second-by-second records from a .fit or .fit.gz file and computes training metrics."""
    df, lap_count = parse_fit_records(file_path)
    if df.empty:
        return {col: np.nan for col in METRIC_COLS}
    return compute_stream_metrics(
        df,
        lap_count=lap_count,
        effective_ftp=effective_ftp,
        effective_max_hr=effective_max_hr,
    )


def _to_naive_ts(val) -> pd.Timestamp | None:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    ts = pd.to_datetime(val, errors="coerce")
    if pd.isna(ts):
        return None
    if ts.tzinfo is not None:
        ts = ts.tz_localize(None)
    return ts


def run_bulk_export():
    """Processes historical Strava bulk export (activities.csv + local .fit files)."""
    activities_csv = STRAVA_DIR / "activities.csv"
    if not activities_csv.exists():
        raise FileNotFoundError(f"Could not find {activities_csv}")

    base_df = pd.read_csv(activities_csv)
    if "Activity Date" in base_df.columns:
        base_df["_parsed_ts"] = pd.to_datetime(base_df["Activity Date"], errors="coerce")
        base_df = base_df.sort_values("_parsed_ts", kind="mergesort", na_position="last").reset_index(drop=True)
    else:
        base_df["_parsed_ts"] = pd.NaT

    print(f"Loaded {len(base_df)} activities from activities.csv (sorted chronologically). Parsing FIT files...")

    history_records: list[dict] = []
    last_known_ftp: int | None = None
    enriched_rows = []

    for idx, row in base_df.iterrows():
        rel_filename = str(row.get("Filename", "")).strip()
        row_ts = _to_naive_ts(row.get("_parsed_ts"))

        stream_df = pd.DataFrame()
        lap_count = 0
        if rel_filename and rel_filename != "nan":
            target_path = STRAVA_DIR / rel_filename
            if target_path.name.endswith((".fit", ".fit.gz")):
                stream_df, lap_count = parse_fit_records(target_path)

        peaks = extract_stream_peaks(stream_df)
        if _valid_num(peaks.get("clean_max_hr")) is None:
            peaks["max_hr"] = _valid_num(row.get("Max Heart Rate"))

        eff_ftp, eff_max_hr = resolve_dynamic_physiology(
            activity_ts=row_ts,
            history_records=history_records,
            current_peaks=peaks,
            last_known_ftp=last_known_ftp,
        )

        if stream_df.empty:
            fit_metrics = {col: np.nan for col in METRIC_COLS}
            fit_metrics["effective_ftp"] = eff_ftp
            fit_metrics["effective_max_hr"] = eff_max_hr
        else:
            fit_metrics = compute_stream_metrics(
                stream_df,
                lap_count=lap_count,
                effective_ftp=eff_ftp,
                effective_max_hr=eff_max_hr,
                peaks=peaks,
            )

        last_known_ftp = eff_ftp
        history_records.append({
            "ts": row_ts,
            "peak_20m_w": peaks.get("peak_20m_w"),
            "peak_5m_w": peaks.get("peak_5m_w"),
            "clean_max_hr": peaks.get("clean_max_hr"),
            "Max Heart Rate": row.get("Max Heart Rate"),
            "effective_ftp": eff_ftp,
        })
        enriched_rows.append(fit_metrics)

        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1}/{len(base_df)} activities...")

    base_df = base_df.drop(columns=["_parsed_ts"], errors="ignore")
    fit_df = pd.DataFrame(enriched_rows, columns=METRIC_COLS)
    enriched_df = pd.concat([base_df.reset_index(drop=True), fit_df], axis=1)

    final_cols = [c for c in BASE_COLS if c in enriched_df.columns] + METRIC_COLS
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    enriched_df[final_cols].to_csv(OUTPUT_CSV, index=False)
    print(f"Done! Saved bulk enriched dataset to {OUTPUT_CSV} ({len(final_cols)} columns).")


# =====================================================================
# 3. SOURCE ADAPTER B: STRAVA V3 REST API
# =====================================================================
class StravaAPIClient:
    """Handles OAuth2 token management and requests to Strava API v3."""

    TOKEN_URL = "https://www.strava.com/oauth/token"
    BASE_URL = "https://www.strava.com/api/v3"

    def __init__(self, client_id: str, client_secret: str, refresh_token: str, token_file: Path):
        self.token_file = token_file
        self.client_id = client_id
        self.client_secret = client_secret
        self.refresh_token = refresh_token
        self.access_token = None
        self.expires_at = 0
        self._load_cached_tokens()

    def _load_cached_tokens(self):
        if self.token_file.exists():
            data = json.loads(self.token_file.read_text())
            self.client_id = self.client_id or data.get("client_id", "")
            self.client_secret = self.client_secret or data.get("client_secret", "")
            self.refresh_token = data.get("refresh_token", self.refresh_token)
            self.access_token = data.get("access_token")
            self.expires_at = data.get("expires_at", 0)

    def _save_tokens(self, token_data: dict):
        payload = {
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "refresh_token": token_data["refresh_token"],
            "access_token": token_data["access_token"],
            "expires_at": token_data["expires_at"],
        }
        self.token_file.write_text(json.dumps(payload, indent=2))
        self.refresh_token = payload["refresh_token"]
        self.access_token = payload["access_token"]
        self.expires_at = payload["expires_at"]

    def ensure_access_token(self) -> str:
        if self.access_token and time.time() < (self.expires_at - 60):
            return self.access_token

        if not all([self.client_id, self.client_secret, self.refresh_token]):
            raise ValueError(
                "Missing Strava API credentials. Set STRAVA_CLIENT_ID, STRAVA_CLIENT_SECRET, "
                "and STRAVA_REFRESH_TOKEN environment variables or populate .strava_tokens.json."
            )

        resp = requests.post(
            self.TOKEN_URL,
            data={
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "grant_type": "refresh_token",
                "refresh_token": self.refresh_token,
            },
            timeout=15,
        )
        if not resp.ok:
            raise requests.HTTPError(
                f"{resp.status_code} {resp.reason} for {self.TOKEN_URL}: {resp.text[:300]}",
                response=resp,
            )
        self._save_tokens(resp.json())
        return self.access_token

    def get(self, endpoint: str, params: dict = None):
        token = self.ensure_access_token()
        url = f"{self.BASE_URL}{endpoint}"
        resp = requests.get(
            url,
            headers={"Authorization": f"Bearer {token}"},
            params=params,
            timeout=20,
        )
        if resp.status_code == 404:
            return None
        if not resp.ok:
            raise requests.HTTPError(
                f"{resp.status_code} {resp.reason} for {url}: {resp.text[:300]}",
                response=resp,
            )
        return resp.json()


def normalize_activity_type(api_type: str) -> str:
    """Maps Strava API camelCase types to match Strava bulk export CSV labels."""
    mapping = {
        "VirtualRide": "Virtual Ride",
        "VirtualRun": "Virtual Run",
        "EBikeRide": "E-Bike Ride",
        "WeightTraining": "Weight Training",
        "Workout": "Workout",
        "Ride": "Ride",
        "Run": "Run",
        "Swim": "Swim",
        "Walk": "Walk",
        "Hike": "Hike",
    }
    return mapping.get(api_type, api_type)


def format_strava_csv_date(iso_date_str: str) -> str:
    """Formats ISO 8601 UTC date ('2026-11-18T18:30:00Z') to Strava CSV format ('Nov 18, 2026, 6:30:00 PM')."""
    if not iso_date_str:
        return ""
    try:
        dt = datetime.fromisoformat(iso_date_str.replace("Z", "+00:00"))
        # Format without leading zero on hour to match Strava export
        hour_12 = dt.strftime("%I").lstrip("0") or "12"
        return f"{dt.strftime('%b %d, %Y')}, {hour_12}:{dt.strftime('%M:%S %p')}"
    except Exception:
        return iso_date_str


def streams_to_dataframe(streams_json: dict) -> pd.DataFrame:
    """Converts Strava API streams dictionary into the standardized 1Hz DataFrame."""
    if not streams_json:
        return pd.DataFrame()

    # Determine stream length from 'time' or first available key
    first_key = "time" if "time" in streams_json else next(iter(streams_json), None)
    if not first_key:
        return pd.DataFrame()

    n_samples = len(streams_json[first_key].get("data", []))
    if n_samples == 0:
        return pd.DataFrame()

    def extract_col(key: str):
        if key in streams_json and "data" in streams_json[key]:
            return streams_json[key]["data"]
        return [np.nan] * n_samples

    return pd.DataFrame({
        "hr": extract_col("heartrate"),
        "power": extract_col("watts"),
        "cadence": extract_col("cadence"),
        "speed": extract_col("velocity_smooth"),
        "altitude": extract_col("altitude"),
    })


def sync_recent_from_api(limit: int = 30, after_days: int = None):
    """Fetches recent activities from Strava API, enriches them, and appends new rows to OUTPUT_CSV."""
    client = StravaAPIClient(
        client_id=STRAVA_CLIENT_ID,
        client_secret=STRAVA_CLIENT_SECRET,
        refresh_token=STRAVA_REFRESH_TOKEN,
        token_file=TOKEN_FILE,
    )

    existing_df = pd.DataFrame()
    existing_ids = set()
    if OUTPUT_CSV.exists():
        existing_df = pd.read_csv(OUTPUT_CSV)
        if "Activity ID" in existing_df.columns:
            existing_ids = set(pd.to_numeric(existing_df["Activity ID"], errors="coerce").dropna().astype(int))
        print(f"Loaded existing dataset ({len(existing_df)} rows). Checking Strava API for new activities...")
    else:
        print("No existing dataset found at OUTPUT_CSV. Creating new dataset from Strava API...")

    params = {"per_page": min(limit, 200), "page": 1}
    if after_days:
        params["after"] = int(time.time() - (after_days * 86400))

    summaries = client.get("/athlete/activities", params=params) or []
    new_summaries = [a for a in summaries if int(a["id"]) not in existing_ids]

    if not new_summaries:
        print("All recent Strava activities are already present in the dataset.")
        return {"fetched": 0, "db_inserted": 0, "total_csv_rows": len(existing_df)}

    # Process oldest-to-newest so chronological order is preserved when appending
    new_summaries.sort(key=lambda x: x.get("start_date", ""))
    print(f"Found {len(new_summaries)} new activities to enrich.")

    history_records: list[dict] = []
    last_known_ftp: int | None = None
    if not existing_df.empty and "Activity Date" in existing_df.columns:
        hist_df = existing_df.copy()
        hist_df["_ts"] = pd.to_datetime(hist_df["Activity Date"], errors="coerce")
        hist_df = hist_df.sort_values("_ts", kind="mergesort", na_position="last")
        for _, hrow in hist_df.iterrows():
            hts = _to_naive_ts(hrow.get("_ts"))
            h_ftp = _valid_num(hrow.get("effective_ftp"))
            if h_ftp and h_ftp > 0:
                last_known_ftp = int(round(h_ftp))
            history_records.append({
                "ts": hts,
                "peak_20m_w": hrow.get("peak_20m_w"),
                "peak_5m_w": hrow.get("peak_5m_w"),
                "Max Heart Rate": hrow.get("Max Heart Rate"),
                "effective_ftp": h_ftp,
            })

    new_rows = []
    for idx, summary in enumerate(new_summaries, start=1):
        act_id = int(summary["id"])
        act_name = summary.get("name", "Unnamed Activity")
        print(f"[{idx}/{len(new_summaries)}] Fetching & enriching Activity {act_id}: {act_name}...")

        # 1. Fetch detailed activity object (includes description, laps, and suffer_score)
        detail = client.get(f"/activities/{act_id}") or summary
        laps = detail.get("laps") or []
        lap_count = len(laps) if laps else 1

        # 2. Fetch 1Hz sensor streams
        streams = client.get(
            f"/activities/{act_id}/streams",
            params={
                "keys": "time,heartrate,watts,cadence,velocity_smooth,altitude",
                "key_by_type": "true",
            },
        )
        stream_df = streams_to_dataframe(streams)
        peaks = extract_stream_peaks(stream_df)
        if _valid_num(peaks.get("clean_max_hr")) is None:
            peaks["max_hr"] = _valid_num(detail.get("max_heartrate"))

        act_ts = _to_naive_ts(detail.get("start_date"))
        eff_ftp, eff_max_hr = resolve_dynamic_physiology(
            activity_ts=act_ts,
            history_records=history_records,
            current_peaks=peaks,
            last_known_ftp=last_known_ftp,
        )
        metrics = compute_stream_metrics(
            stream_df,
            lap_count=lap_count,
            effective_ftp=eff_ftp,
            effective_max_hr=eff_max_hr,
            peaks=peaks,
        )

        last_known_ftp = eff_ftp
        history_records.append({
            "ts": act_ts,
            "peak_20m_w": peaks.get("peak_20m_w"),
            "peak_5m_w": peaks.get("peak_5m_w"),
            "clean_max_hr": peaks.get("clean_max_hr"),
            "Max Heart Rate": detail.get("max_heartrate"),
            "effective_ftp": eff_ftp,
        })

        # 3. Map base metadata to match Strava bulk export CSV columns
        base_metadata = {
            "Activity ID": act_id,
            "Activity Date": format_strava_csv_date(detail.get("start_date")),
            "Activity Name": act_name,
            "Activity Type": normalize_activity_type(detail.get("sport_type") or detail.get("type", "")),
            "Activity Description": detail.get("description") or "",
            "Elapsed Time": detail.get("elapsed_time", np.nan),
            "Moving Time": detail.get("moving_time", np.nan),
            "Distance": round(detail.get("distance", 0.0) / 1000.0, 2),  # Convert meters to km
            "Elevation Gain": detail.get("total_elevation_gain", np.nan),
            "Average Heart Rate": detail.get("average_heartrate", np.nan),
            "Max Heart Rate": detail.get("max_heartrate", np.nan),
            "Average Watts": detail.get("average_watts", np.nan),
            "Weighted Average Power": detail.get("weighted_average_watts", np.nan),
            "Relative Effort": detail.get("suffer_score", np.nan),
        }

        row_data = {**base_metadata, **metrics}
        new_rows.append(row_data)

    new_df = pd.DataFrame(new_rows)
    final_cols = BASE_COLS + METRIC_COLS

    if not existing_df.empty:
        # Align columns with existing dataset
        for col in final_cols:
            if col not in existing_df.columns:
                existing_df[col] = np.nan
        combined_df = pd.concat([existing_df[final_cols], new_df[final_cols]], ignore_index=True)
    else:
        combined_df = new_df[final_cols]

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(OUTPUT_CSV, index=False)
    db_inserted = app_db.insert_activities(new_rows)
    print(
        f"Appended {len(new_df)} new activities to {OUTPUT_CSV} "
        f"(Total rows: {len(combined_df)}, DB inserted: {db_inserted})."
    )
    return {
        "source": "strava",
        "fetched": len(new_df),
        "db_inserted": db_inserted,
        "total_csv_rows": len(combined_df),
    }


# =====================================================================
# 4. SOURCE ADAPTER C: INTERVALS.ICU REST API
# =====================================================================
class IntervalsAPIClient:
    """Official Intervals.icu v1 REST API client using HTTP Basic Auth ('API_KEY': api_key)."""

    BASE_URL = "https://intervals.icu/api/v1"

    def __init__(self, api_key: str, athlete_id: str = "0"):
        self.api_key = (api_key or "").strip()
        self.athlete_id = (athlete_id or "0").strip() or "0"
        if not self.api_key:
            raise ValueError(
                "Missing INTERVALS_API_KEY. Generate an API key in Intervals.icu "
                "(Settings -> Developer Settings) and set INTERVALS_API_KEY in your environment or .env."
            )
        self.auth = ("API_KEY", self.api_key)

    def get_json(self, endpoint: str, params: dict | None = None):
        url = f"{self.BASE_URL}{endpoint}"
        resp = requests.get(url, auth=self.auth, params=params, timeout=25)
        if resp.status_code == 404:
            return None
        if not resp.ok:
            raise requests.HTTPError(
                f"{resp.status_code} {resp.reason} for {url}: {resp.text[:300]}",
                response=resp,
            )
        return resp.json()

    def get_bytes(self, endpoint: str, params: dict | None = None) -> bytes | None:
        url = f"{self.BASE_URL}{endpoint}"
        resp = requests.get(url, auth=self.auth, params=params, timeout=30)
        if resp.status_code in (404, 204):
            return None
        if not resp.ok:
            raise requests.HTTPError(
                f"{resp.status_code} {resp.reason} for {url}: {resp.text[:300]}",
                response=resp,
            )
        return resp.content


def intervals_id_to_int(raw_id: str | int) -> int:
    """
    Maps an Intervals.icu activity ID (e.g. 'i55751783' or numeric string) to a deterministic
    64-bit positive integer for SQLite's INTEGER PRIMARY KEY without colliding with Strava IDs.
    """
    text = str(raw_id).strip()
    if text.isdigit():
        return int(text)
    if text.startswith("i") and text[1:].isdigit():
        return 9_000_000_000_000 + int(text[1:])
    return 8_000_000_000_000 + int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:5], "big")


def intervals_streams_to_dataframe(streams_payload) -> pd.DataFrame:
    """Converts Intervals.icu /activity/{id}/streams.json response into the standardized 1Hz DataFrame."""
    if not streams_payload:
        return pd.DataFrame()

    # Intervals.icu returns either a list of {"type": ..., "data": [...]} or a dict keyed by stream type
    if isinstance(streams_payload, list):
        by_type = {
            item.get("type"): item.get("data", [])
            for item in streams_payload
            if isinstance(item, dict) and item.get("type")
        }
    elif isinstance(streams_payload, dict):
        by_type = {
            k: (v.get("data", []) if isinstance(v, dict) else v)
            for k, v in streams_payload.items()
        }
    else:
        return pd.DataFrame()

    if not by_type:
        return pd.DataFrame()

    n_samples = len(by_type.get("time") or next(iter(by_type.values()), []))
    if n_samples == 0:
        return pd.DataFrame()

    def extract_col(*keys: str):
        for k in keys:
            vals = by_type.get(k)
            if vals and len(vals) == n_samples:
                return vals
        return [np.nan] * n_samples

    return pd.DataFrame({
        "hr": extract_col("heartrate"),
        "power": extract_col("watts"),
        "cadence": extract_col("cadence"),
        "speed": extract_col("velocity_smooth"),
        "altitude": extract_col("altitude", "fixed_altitude"),
    })


def _is_duplicate_timestamp(
    candidate_ts: pd.Timestamp | None,
    existing_ts_list: list[pd.Timestamp],
    tolerance_sec: int = 90,
) -> bool:
    """Returns True if candidate_ts falls within tolerance_sec of any timestamp in existing_ts_list."""
    if candidate_ts is None or pd.isna(candidate_ts):
        return False
    for existing_ts in existing_ts_list:
        if abs((candidate_ts - existing_ts).total_seconds()) <= tolerance_sec:
            return True
    return False


def sync_recent_from_intervals(limit: int = 30, after_days: int | None = None) -> dict:
    """
    Fetches recent activities from Intervals.icu, downloads processed/original .fit files
    (falling back to streams if needed), enriches them via the shared metric engine,
    appends new rows to OUTPUT_CSV, and inserts them into SQLite.
    Deduplicates against both Activity IDs and start timestamps (+/-90s) so historical
    Strava rides are never duplicated.
    """
    api_key = INTERVALS_API_KEY or os.getenv("INTERVALS_API_KEY", "")
    athlete_id = INTERVALS_ATHLETE_ID or os.getenv("INTERVALS_ATHLETE_ID", "0")
    client = IntervalsAPIClient(api_key=api_key, athlete_id=athlete_id)

    existing_df = pd.DataFrame()
    existing_ids: set[int] = set()
    existing_timestamps: list[pd.Timestamp] = []
    if OUTPUT_CSV.exists():
        existing_df = pd.read_csv(OUTPUT_CSV)
        if "Activity ID" in existing_df.columns:
            existing_ids = set(pd.to_numeric(existing_df["Activity ID"], errors="coerce").dropna().astype(int))
        if "Activity Date" in existing_df.columns:
            parsed_dates = pd.to_datetime(existing_df["Activity Date"], errors="coerce").dropna()
            existing_timestamps = [ts.tz_localize(None) if ts.tzinfo is not None else ts for ts in parsed_dates]
        print(f"Loaded existing dataset ({len(existing_df)} rows). Checking Intervals.icu for new activities...")
    else:
        print("No existing dataset found at OUTPUT_CSV. Creating new dataset from Intervals.icu...")

    lookback_days = max(1, int(after_days)) if after_days else 90
    newest_date = (date.today() + timedelta(days=1)).isoformat()
    oldest_date = (date.today() - timedelta(days=lookback_days)).isoformat()

    raw_activities = client.get_json(
        f"/athlete/{client.athlete_id}/activities",
        params={"oldest": oldest_date, "newest": newest_date},
    ) or []

    skipped_strava_stubs = 0
    candidates = []
    for act in raw_activities:
        if not isinstance(act, dict) or not act.get("id"):
            continue
        start_iso = act.get("start_date") or act.get("start_date_local")
        # Intervals.icu returns minimal stub objects (no start_date/name) for Strava-sourced activities
        if not start_iso:
            if str(act.get("source", "")).upper() == "STRAVA":
                skipped_strava_stubs += 1
            continue
        candidates.append(act)

    # Sort newest-first to respect `limit`, then filter out duplicates
    candidates.sort(key=lambda x: x.get("start_date") or x.get("start_date_local") or "", reverse=True)
    candidates = candidates[: max(1, min(int(limit), 200))]

    new_summaries = []
    for act in candidates:
        int_id = intervals_id_to_int(act["id"])
        if int_id in existing_ids:
            continue
        start_iso = act.get("start_date") or act.get("start_date_local")
        act_ts = _to_naive_ts(start_iso)
        if _is_duplicate_timestamp(act_ts, existing_timestamps, tolerance_sec=90):
            continue
        new_summaries.append(act)

    if not new_summaries:
        print(
            f"All recent Intervals.icu activities are already present in the dataset "
            f"(skipped_strava_stubs={skipped_strava_stubs})."
        )
        return {
            "source": "intervals",
            "fetched": 0,
            "db_inserted": 0,
            "total_csv_rows": len(existing_df),
            "skipped_strava_stubs": skipped_strava_stubs,
        }

    # Process oldest-to-newest so rolling 90d FTP and chronological order are preserved
    new_summaries.sort(key=lambda x: x.get("start_date") or x.get("start_date_local") or "")
    print(f"Found {len(new_summaries)} new Intervals.icu activities to enrich.")

    history_records: list[dict] = []
    last_known_ftp: int | None = None
    if not existing_df.empty and "Activity Date" in existing_df.columns:
        hist_df = existing_df.copy()
        hist_df["_ts"] = pd.to_datetime(hist_df["Activity Date"], errors="coerce")
        hist_df = hist_df.sort_values("_ts", kind="mergesort", na_position="last")
        for _, hrow in hist_df.iterrows():
            hts = _to_naive_ts(hrow.get("_ts"))
            h_ftp = _valid_num(hrow.get("effective_ftp"))
            if h_ftp and h_ftp > 0:
                last_known_ftp = int(round(h_ftp))
            history_records.append({
                "ts": hts,
                "peak_20m_w": hrow.get("peak_20m_w"),
                "peak_5m_w": hrow.get("peak_5m_w"),
                "Max Heart Rate": hrow.get("Max Heart Rate"),
                "effective_ftp": h_ftp,
            })

    new_rows = []
    for idx, summary in enumerate(new_summaries, start=1):
        raw_id = str(summary["id"])
        int_id = intervals_id_to_int(raw_id)
        act_name = summary.get("name") or "Unnamed Activity"
        print(f"[{idx}/{len(new_summaries)}] Fetching & enriching Intervals.icu Activity {raw_id}: {act_name}...")

        stream_df = pd.DataFrame()
        lap_count = 0

        # 1. Prefer Intervals.icu processed FIT file (.fit.gz), then original file, then JSON streams
        for file_endpoint in (f"/activity/{raw_id}/fit-file", f"/activity/{raw_id}/file"):
            try:
                raw_bytes = client.get_bytes(file_endpoint)
                if raw_bytes:
                    stream_df, lap_count = parse_fit_bytes(raw_bytes, source_name=f"intervals_{raw_id}")
                    if not stream_df.empty:
                        break
            except Exception as exc:
                print(f"Warning: Could not download {file_endpoint}: {exc}")

        if stream_df.empty:
            try:
                streams_payload = client.get_json(
                    f"/activity/{raw_id}/streams.json",
                    params={"types": "time,heartrate,watts,cadence,velocity_smooth,altitude"},
                )
                stream_df = intervals_streams_to_dataframe(streams_payload)
            except Exception as exc:
                print(f"Warning: Could not fetch streams for Intervals.icu activity {raw_id}: {exc}")

        if lap_count == 0:
            intervals_list = summary.get("icu_intervals") or []
            lap_count = len(intervals_list) if intervals_list else 1

        peaks = extract_stream_peaks(stream_df)
        if _valid_num(peaks.get("clean_max_hr")) is None:
            peaks["max_hr"] = _valid_num(summary.get("max_heartrate"))

        start_iso = summary.get("start_date") or summary.get("start_date_local")
        act_ts = _to_naive_ts(start_iso)
        eff_ftp, eff_max_hr = resolve_dynamic_physiology(
            activity_ts=act_ts,
            history_records=history_records,
            current_peaks=peaks,
            last_known_ftp=last_known_ftp,
        )
        metrics = compute_stream_metrics(
            stream_df,
            lap_count=lap_count,
            effective_ftp=eff_ftp,
            effective_max_hr=eff_max_hr,
            peaks=peaks,
        )

        last_known_ftp = eff_ftp
        history_records.append({
            "ts": act_ts,
            "peak_20m_w": peaks.get("peak_20m_w"),
            "peak_5m_w": peaks.get("peak_5m_w"),
            "clean_max_hr": peaks.get("clean_max_hr"),
            "Max Heart Rate": summary.get("max_heartrate"),
            "effective_ftp": eff_ftp,
        })
        if act_ts is not None:
            existing_timestamps.append(act_ts)

        distance_m = _valid_num(summary.get("distance")) or 0.0
        avg_watts = _valid_num(summary.get("icu_average_watts"))
        if avg_watts is None:
            avg_watts = _valid_num(summary.get("average_watts"))
        if avg_watts is None:
            avg_watts = metrics.get("avg_power_fit", np.nan)

        weighted_watts = _valid_num(summary.get("icu_weighted_avg_watts"))
        if weighted_watts is None:
            weighted_watts = _valid_num(summary.get("weighted_average_watts"))
        if weighted_watts is None:
            weighted_watts = metrics.get("normalized_power", np.nan)

        base_metadata = {
            "Activity ID": int_id,
            "Activity Date": format_strava_csv_date(start_iso),
            "Activity Name": act_name,
            "Activity Type": normalize_activity_type(summary.get("type") or ""),
            "Activity Description": summary.get("description") or "",
            "Elapsed Time": summary.get("elapsed_time") if summary.get("elapsed_time") is not None else summary.get("moving_time", np.nan),
            "Moving Time": summary.get("moving_time", np.nan),
            "Distance": round(distance_m / 1000.0, 2),
            "Elevation Gain": summary.get("total_elevation_gain", np.nan),
            "Average Heart Rate": summary.get("average_heartrate", np.nan),
            "Max Heart Rate": summary.get("max_heartrate", np.nan),
            "Average Watts": avg_watts,
            "Weighted Average Power": weighted_watts,
            "Relative Effort": summary.get("icu_training_load", np.nan),
        }

        row_data = {**base_metadata, **metrics}
        new_rows.append(row_data)

    new_df = pd.DataFrame(new_rows)
    final_cols = BASE_COLS + METRIC_COLS

    if not existing_df.empty:
        for col in final_cols:
            if col not in existing_df.columns:
                existing_df[col] = np.nan
        combined_df = pd.concat([existing_df[final_cols], new_df[final_cols]], ignore_index=True)
    else:
        combined_df = new_df[final_cols]

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(OUTPUT_CSV, index=False)
    db_inserted = app_db.insert_activities(new_rows)
    print(
        f"Appended {len(new_df)} new Intervals.icu activities to {OUTPUT_CSV} "
        f"(Total rows: {len(combined_df)}, DB inserted: {db_inserted})."
    )
    return {
        "source": "intervals",
        "fetched": len(new_df),
        "db_inserted": db_inserted,
        "total_csv_rows": len(combined_df),
        "skipped_strava_stubs": skipped_strava_stubs,
    }


# =====================================================================
# 5. CLI ENTRY POINT
# =====================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Enrich cycling activities from bulk FIT export, Intervals.icu API, or Strava API."
    )
    default_mode = (
        "intervals"
        if (OUTPUT_CSV.exists() and (INTERVALS_API_KEY or os.getenv("INTERVALS_API_KEY")))
        else ("api" if OUTPUT_CSV.exists() else "bulk")
    )
    parser.add_argument(
        "--mode",
        choices=["bulk", "intervals", "strava", "api"],
        default=default_mode,
        help=(
            "'bulk' parses strava_export/activities.csv + FIT files; "
            "'intervals' fetches recent activities from Intervals.icu API; "
            "'strava' / 'api' fetches recent rides from Strava API."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=30,
        help="Max number of recent activities to check in API mode (default: 30).",
    )
    parser.add_argument(
        "--after-days",
        type=int,
        default=None,
        help="Only check activities from the last N days (Intervals.icu defaults to 90 days).",
    )
    args = parser.parse_args()

    if args.mode == "bulk":
        run_bulk_export()
    elif args.mode == "intervals":
        sync_recent_from_intervals(limit=args.limit, after_days=args.after_days)
    else:
        sync_recent_from_api(limit=args.limit, after_days=args.after_days)


if __name__ == "__main__":
    main()