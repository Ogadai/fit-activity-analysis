"""SQLite database engine, CSV seeder, and CTL/ATL/TSB training load calculations."""

from __future__ import annotations

import math
import sqlite3
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from src import config

BASE_COLS: list[str] = [
    "Activity ID",
    "Activity Date",
    "Activity Name",
    "Activity Type",
    "Activity Description",
    "Elapsed Time",
    "Moving Time",
    "Distance",
    "Elevation Gain",
    "Average Heart Rate",
    "Max Heart Rate",
    "Average Watts",
    "Weighted Average Power",
    "Relative Effort",
]

METRIC_COLS: list[str] = [
    "fit_lap_count",
    "hr_z1_pct",
    "hr_z2_pct",
    "hr_z3_pct",
    "hr_z4_pct",
    "hr_z5_pct",
    "avg_power_fit",
    "normalized_power",
    "intensity_factor",
    "variability_index",
    "pwr_z1_pct",
    "pwr_z2_pct",
    "pwr_z3_pct",
    "pwr_z4_pct",
    "pwr_z5_plus_pct",
    "peak_1m_w",
    "peak_5m_w",
    "peak_20m_w",
    "hiit_surges_count",
    "aerobic_decoupling_pct",
    "climbing_time_pct",
]

DERIVED_COLS: list[str] = [
    "activity_timestamp",
    "activity_date_iso",
    "tss",
]

ALL_COLS: list[str] = BASE_COLS + METRIC_COLS + DERIVED_COLS

_INT_COLS = {"Activity ID", "fit_lap_count", "hiit_surges_count"}
_TEXT_COLS = {
    "Activity Date",
    "Activity Name",
    "Activity Type",
    "Activity Description",
    "activity_timestamp",
    "activity_date_iso",
}


def get_connection(db_path: Path | None = None) -> sqlite3.Connection:
    """Open a SQLite connection with Row factory enabled."""
    target = Path(db_path or config.DB_PATH)
    target.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(target))
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Path | None = None) -> None:
    """Create the activities table and date index if they do not exist."""
    col_defs: list[str] = []
    for col in ALL_COLS:
        if col == "Activity ID":
            col_defs.append(f'"{col}" INTEGER PRIMARY KEY')
        elif col in _INT_COLS:
            col_defs.append(f'"{col}" INTEGER')
        elif col in _TEXT_COLS:
            col_defs.append(f'"{col}" TEXT')
        else:
            col_defs.append(f'"{col}" REAL')

    ddl = f"""
    CREATE TABLE IF NOT EXISTS activities (
        {", ".join(col_defs)}
    );
    CREATE INDEX IF NOT EXISTS idx_activities_date_iso
        ON activities (activity_date_iso);
    """
    with get_connection(db_path) as conn:
        conn.executescript(ddl)


def _clean_value(val: Any, col: str) -> Any:
    """Normalize pandas/numpy values to native Python types or None for SQLite."""
    if val is None:
        return None
    if isinstance(val, float) and math.isnan(val):
        return None
    try:
        if pd.isna(val):
            return None
    except (TypeError, ValueError):
        pass

    if col in _INT_COLS:
        return int(float(val))
    if col in _TEXT_COLS:
        return str(val)
    return float(val)


def parse_activity_timestamp(raw_date: Any) -> tuple[str | None, str | None]:
    """Parse an activity date string into (ISO-8601 timestamp, YYYY-MM-DD)."""
    if raw_date is None or (isinstance(raw_date, float) and math.isnan(raw_date)):
        return None, None
    text = str(raw_date).strip()
    if not text or text.lower() == "nan":
        return None, None

    dt = pd.to_datetime(text, errors="coerce")
    if pd.isna(dt):
        return None, None

    return dt.isoformat(), dt.strftime("%Y-%m-%d")


def compute_tss(row: dict[str, Any], ftp: int | None = None) -> float | None:
    """Compute Coggan Training Stress Score (TSS) from activity power and duration."""
    effective_ftp = ftp or config.FTP
    if effective_ftp <= 0:
        return None

    moving_time = _clean_value(row.get("Moving Time"), "Moving Time")
    elapsed_time = _clean_value(row.get("Elapsed Time"), "Elapsed Time")
    duration_sec = moving_time if (moving_time and moving_time > 0) else elapsed_time
    if not duration_sec or duration_sec <= 0:
        return None

    np_val = (
        _clean_value(row.get("normalized_power"), "normalized_power")
        or _clean_value(row.get("Weighted Average Power"), "Weighted Average Power")
        or _clean_value(row.get("Average Watts"), "Average Watts")
    )
    if not np_val or np_val <= 0:
        return None

    if_val = _clean_value(row.get("intensity_factor"), "intensity_factor")
    if not if_val or if_val <= 0:
        if_val = np_val / float(effective_ftp)

    tss = (duration_sec * np_val * if_val) / (float(effective_ftp) * 3600.0) * 100.0
    return round(tss, 1)


def _prepare_row_tuple(raw_row: dict[str, Any]) -> tuple[Any, ...] | None:
    """Enrich a raw activity dictionary and return an ordered tuple for INSERT."""
    act_id = _clean_value(raw_row.get("Activity ID"), "Activity ID")
    if act_id is None:
        return None

    enriched = dict(raw_row)
    ts_iso, date_iso = parse_activity_timestamp(enriched.get("Activity Date"))
    enriched["activity_timestamp"] = enriched.get("activity_timestamp") or ts_iso
    enriched["activity_date_iso"] = enriched.get("activity_date_iso") or date_iso
    if _clean_value(enriched.get("tss"), "tss") is None:
        enriched["tss"] = compute_tss(enriched)

    return tuple(_clean_value(enriched.get(col), col) for col in ALL_COLS)


def insert_activities(rows: list[dict[str, Any]], db_path: Path | None = None) -> int:
    """Enrich and insert activity rows into SQLite using INSERT OR IGNORE."""
    if not rows:
        return 0

    init_db(db_path)
    quoted_cols = ", ".join(f'"{col}"' for col in ALL_COLS)
    placeholders = ", ".join("?" for _ in ALL_COLS)
    sql = f'INSERT OR IGNORE INTO activities ({quoted_cols}) VALUES ({placeholders})'

    tuples = []
    for row in rows:
        prepared = _prepare_row_tuple(row)
        if prepared is not None:
            tuples.append(prepared)

    if not tuples:
        return 0

    with get_connection(db_path) as conn:
        before = conn.total_changes
        conn.executemany(sql, tuples)
        conn.commit()
        return conn.total_changes - before


def seed_from_csv(
    csv_path: Path | None = None,
    db_path: Path | None = None,
) -> int:
    """Seed cycling.db with any activities from enriched_activities.csv not yet in DB."""
    init_db(db_path)
    target_csv = Path(csv_path or config.CSV_PATH)
    if not target_csv.exists():
        return 0

    df = pd.read_csv(target_csv)
    if df.empty or "Activity ID" not in df.columns:
        return 0

    with get_connection(db_path) as conn:
        existing_ids = {
            row[0]
            for row in conn.execute('SELECT "Activity ID" FROM activities').fetchall()
        }

    df["_act_id_num"] = pd.to_numeric(df["Activity ID"], errors="coerce")
    df = df.dropna(subset=["_act_id_num"])
    new_df = df[~df["_act_id_num"].astype(int).isin(existing_ids)].drop(columns=["_act_id_num"])

    if new_df.empty:
        return 0

    return insert_activities(new_df.to_dict(orient="records"), db_path=db_path)


def compute_training_load(
    db_path: Path | None = None,
    as_of_date: str | None = None,
) -> dict[str, Any]:
    """
    Build a continuous daily date series from the first recorded activity up to today,
    aggregate daily TSS, and calculate CTL (42d), ATL (7d), TSB (CTL - ATL),
    and 90-day rolling FTP (0.95 * max(peak_20m_w)).
    """
    init_db(db_path)
    end_dt = date.fromisoformat(as_of_date) if as_of_date else date.today()

    with get_connection(db_path) as conn:
        rows = conn.execute(
            """
            SELECT activity_date_iso, tss, peak_20m_w
            FROM activities
            WHERE activity_date_iso IS NOT NULL
            ORDER BY activity_date_iso ASC
            """
        ).fetchall()
        total_activities = conn.execute("SELECT COUNT(*) FROM activities").fetchone()[0]

    if not rows:
        return {
            "as_of_date": end_dt.isoformat(),
            "ctl": 0.0,
            "atl": 0.0,
            "tsb": 0.0,
            "rolling_90d_ftp": None,
            "rolling_90d_peak_20m_w": None,
            "configured_ftp": config.FTP,
            "total_activities": total_activities,
            "history": [],
        }

    df = pd.DataFrame([dict(r) for r in rows])
    df["date"] = pd.to_datetime(df["activity_date_iso"]).dt.date
    df["tss"] = pd.to_numeric(df["tss"], errors="coerce").fillna(0.0)
    df["peak_20m_w"] = pd.to_numeric(df["peak_20m_w"], errors="coerce")

    start_dt = df["date"].min()
    if end_dt < start_dt:
        end_dt = start_dt

    daily_tss_series = df.groupby("date")["tss"].sum()

    ctl = 0.0
    atl = 0.0
    history: list[dict[str, Any]] = []
    current_dt = start_dt
    while current_dt <= end_dt:
        day_tss = float(daily_tss_series.get(current_dt, 0.0))
        ctl = ctl + (day_tss - ctl) / 42.0
        atl = atl + (day_tss - atl) / 7.0
        tsb = ctl - atl
        history.append(
            {
                "date": current_dt.isoformat(),
                "tss": round(day_tss, 1),
                "ctl": round(ctl, 1),
                "atl": round(atl, 1),
                "tsb": round(tsb, 1),
            }
        )
        current_dt += timedelta(days=1)

    window_start = end_dt - timedelta(days=89)
    recent_peaks = df[(df["date"] >= window_start) & (df["date"] <= end_dt)]["peak_20m_w"].dropna()
    if not recent_peaks.empty:
        peak_20m = round(float(recent_peaks.max()), 1)
        rolling_90d_ftp = round(0.95 * peak_20m, 1)
    else:
        peak_20m = None
        rolling_90d_ftp = None

    latest = history[-1]
    return {
        "as_of_date": end_dt.isoformat(),
        "ctl": latest["ctl"],
        "atl": latest["atl"],
        "tsb": latest["tsb"],
        "rolling_90d_ftp": rolling_90d_ftp,
        "rolling_90d_peak_20m_w": peak_20m,
        "configured_ftp": config.FTP,
        "total_activities": total_activities,
        "history": history,
    }
