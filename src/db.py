"""SQLite database engine, CSV seeder, and CTL/ATL/TSB training load calculations."""

from __future__ import annotations

import contextlib
import json
import math
import sqlite3
from datetime import date, datetime, timedelta, timezone
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
    "effective_ftp",
    "effective_max_hr",
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

_INT_COLS = {
    "Activity ID",
    "fit_lap_count",
    "effective_ftp",
    "effective_max_hr",
    "hiit_surges_count",
}
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

    CREATE TABLE IF NOT EXISTS training_plans (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        goal TEXT,
        start_date TEXT,
        end_date TEXT,
        plan_content TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'active',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_training_plans_status
        ON training_plans (status, updated_at);

    CREATE TABLE IF NOT EXISTS athlete_notes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        category TEXT NOT NULL DEFAULT 'general',
        content TEXT NOT NULL,
        effective_from TEXT,
        effective_until TEXT,
        is_active INTEGER NOT NULL DEFAULT 1,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_athlete_notes_active
        ON athlete_notes (is_active, effective_until);

    CREATE TABLE IF NOT EXISTS activity_timelines (
        activity_id INTEGER PRIMARY KEY,
        source TEXT,
        duration_sec INTEGER,
        segment_count INTEGER NOT NULL DEFAULT 0,
        summary_text TEXT NOT NULL,
        segments_json TEXT NOT NULL,
        highlights_json TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        FOREIGN KEY (activity_id) REFERENCES activities ("Activity ID") ON DELETE CASCADE
    );
    """
    with contextlib.closing(get_connection(db_path)) as conn:
        conn.executescript(ddl)
        existing_cols = {
            row[1] for row in conn.execute("PRAGMA table_info(activities)").fetchall()
        }
        for col_name in ("effective_ftp", "effective_max_hr"):
            if col_name not in existing_cols:
                conn.execute(f'ALTER TABLE activities ADD COLUMN "{col_name}" INTEGER;')
        conn.commit()


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

    if col in _TEXT_COLS:
        return str(val)

    if isinstance(val, str):
        val = val.strip().replace(",", "")
        if not val or val.lower() == "nan":
            return None

    if col in _INT_COLS:
        if isinstance(val, int) and not isinstance(val, bool):
            return val
        try:
            return int(val)
        except (TypeError, ValueError):
            pass

    try:
        num = float(val)
    except (TypeError, ValueError):
        return None
    if math.isnan(num) or math.isinf(num):
        return None

    if col in _INT_COLS:
        return int(num)
    return num


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
    row_ftp = _clean_value(row.get("effective_ftp"), "effective_ftp")
    effective_ftp = ftp or row_ftp or config.DEFAULT_FTP
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

    with contextlib.closing(get_connection(db_path)) as conn:
        before = conn.total_changes
        conn.executemany(sql, tuples)
        conn.commit()
        return conn.total_changes - before


def _backfill_dynamic_physiology(df: pd.DataFrame) -> pd.DataFrame:
    """
    Chronologically backfill missing effective_ftp and effective_max_hr from
    peak_20m_w, peak_5m_w, and Max Heart Rate, and recalculate intensity_factor
    and tss using effective_ftp.
    """
    work = df.copy()
    if "Activity Date" in work.columns:
        work["_ts"] = pd.to_datetime(work["Activity Date"], errors="coerce")
        if work["_ts"].dt.tz is not None:
            work["_ts"] = work["_ts"].dt.tz_localize(None)
        work = work.sort_values("_ts", kind="mergesort", na_position="last").reset_index(drop=True)
    else:
        work["_ts"] = pd.NaT

    history: list[dict[str, Any]] = []
    last_known_ftp: int | None = None

    for idx, row in work.iterrows():
        ts = row["_ts"]
        has_ts = pd.notna(ts)

        existing_eff_ftp = _clean_value(row.get("effective_ftp"), "effective_ftp")
        existing_eff_hr = _clean_value(row.get("effective_max_hr"), "effective_max_hr")

        p20_cur = _clean_value(row.get("peak_20m_w"), "peak_20m_w")
        p5_cur = _clean_value(row.get("peak_5m_w"), "peak_5m_w")
        hr_cur = _clean_value(row.get("Max Heart Rate"), "Max Heart Rate")

        if existing_eff_ftp is None:
            ftp_candidates: list[float] = []
            if has_ts:
                ftp_cutoff = ts - timedelta(days=90)
                for h in history:
                    hts = h["ts"]
                    if hts is not None and ftp_cutoff <= hts <= ts:
                        if h["p20"] and h["p20"] > 0:
                            ftp_candidates.append(0.95 * h["p20"])
                        if h["p5"] and h["p5"] > 0:
                            ftp_candidates.append(0.80 * h["p5"])
            if p20_cur and p20_cur > 0:
                ftp_candidates.append(0.95 * p20_cur)
            if p5_cur and p5_cur > 0:
                ftp_candidates.append(0.80 * p5_cur)

            if ftp_candidates:
                eff_ftp = int(round(max(ftp_candidates)))
            elif last_known_ftp and last_known_ftp > 0:
                eff_ftp = int(last_known_ftp)
            else:
                eff_ftp = config.DEFAULT_FTP

            work.at[idx, "effective_ftp"] = eff_ftp

            np_val = _clean_value(row.get("normalized_power"), "normalized_power")
            if np_val and np_val > 0 and eff_ftp > 0:
                work.at[idx, "intensity_factor"] = round(np_val / float(eff_ftp), 2)

            row_dict = work.loc[idx].to_dict()
            work.at[idx, "tss"] = compute_tss(row_dict, ftp=eff_ftp)
        else:
            eff_ftp = int(existing_eff_ftp)

        if existing_eff_hr is None:
            hr_candidates: list[float] = []
            if has_ts:
                hr_cutoff = ts - timedelta(days=365)
                for h in history:
                    hts = h["ts"]
                    if hts is not None and hr_cutoff <= hts <= ts:
                        if h["hr"] is not None and 135 <= h["hr"] <= 208:
                            hr_candidates.append(h["hr"])
            if hr_cur is not None and 135 <= hr_cur <= 208:
                hr_candidates.append(hr_cur)

            eff_hr = int(round(max(hr_candidates))) if hr_candidates else config.DEFAULT_MAX_HR
            work.at[idx, "effective_max_hr"] = eff_hr

        last_known_ftp = eff_ftp
        history.append({
            "ts": ts if has_ts else None,
            "p20": p20_cur,
            "p5": p5_cur,
            "hr": hr_cur,
        })

    return work.drop(columns=["_ts"], errors="ignore")


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

    needs_backfill = (
        "effective_ftp" not in df.columns
        or "effective_max_hr" not in df.columns
        or df["effective_ftp"].isna().any()
        or df["effective_max_hr"].isna().any()
    )
    if needs_backfill:
        df = _backfill_dynamic_physiology(df)

    with contextlib.closing(get_connection(db_path)) as conn:
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

    with contextlib.closing(get_connection(db_path)) as conn:
        rows = conn.execute(
            """
            SELECT
                "Activity ID",
                "Activity Name",
                activity_date_iso,
                tss,
                peak_20m_w,
                peak_5m_w,
                "Max Heart Rate",
                effective_ftp,
                effective_max_hr
            FROM activities
            WHERE activity_date_iso IS NOT NULL
            ORDER BY activity_date_iso ASC, "Activity ID" ASC
            """
        ).fetchall()
        total_activities = conn.execute("SELECT COUNT(*) FROM activities").fetchone()[0]

    if not rows:
        return {
            "as_of_date": end_dt.isoformat(),
            "ctl": 0.0,
            "atl": 0.0,
            "tsb": 0.0,
            "effective_ftp": config.DEFAULT_FTP,
            "effective_max_hr": config.DEFAULT_MAX_HR,
            "peak_power_90d_source": None,
            "rolling_90d_ftp": None,
            "rolling_90d_peak_20m_w": None,
            "configured_ftp": config.DEFAULT_FTP,
            "total_activities": total_activities,
            "history": [],
        }

    df = pd.DataFrame([dict(r) for r in rows])
    df["date"] = pd.to_datetime(df["activity_date_iso"]).dt.date
    df["tss"] = pd.to_numeric(df["tss"], errors="coerce").fillna(0.0)
    df["peak_20m_w"] = pd.to_numeric(df["peak_20m_w"], errors="coerce")
    df["peak_5m_w"] = pd.to_numeric(df["peak_5m_w"], errors="coerce")
    df["Max Heart Rate"] = pd.to_numeric(df["Max Heart Rate"], errors="coerce")
    df["effective_ftp"] = pd.to_numeric(df["effective_ftp"], errors="coerce")
    df["effective_max_hr"] = pd.to_numeric(df["effective_max_hr"], errors="coerce")

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

    window_90_start = end_dt - timedelta(days=89)
    recent_90_df = df[(df["date"] >= window_90_start) & (df["date"] <= end_dt)]

    best_ftp_est: float | None = None
    peak_power_source: dict[str, Any] | None = None

    for _, r in recent_90_df.iterrows():
        p20 = r["peak_20m_w"]
        p5 = r["peak_5m_w"]
        if pd.notna(p20) and p20 > 0:
            est = 0.95 * float(p20)
            if best_ftp_est is None or est > best_ftp_est:
                best_ftp_est = est
                peak_power_source = {
                    "date": r["activity_date_iso"],
                    "activity_id": int(r["Activity ID"]),
                    "activity_name": r["Activity Name"],
                    "metric": "peak_20m_w",
                    "watts": round(float(p20), 1),
                    "estimated_ftp": int(round(est)),
                }
        if pd.notna(p5) and p5 > 0:
            est = 0.80 * float(p5)
            if best_ftp_est is None or est > best_ftp_est:
                best_ftp_est = est
                peak_power_source = {
                    "date": r["activity_date_iso"],
                    "activity_id": int(r["Activity ID"]),
                    "activity_name": r["Activity Name"],
                    "metric": "peak_5m_w",
                    "watts": round(float(p5), 1),
                    "estimated_ftp": int(round(est)),
                }

    if best_ftp_est is not None:
        effective_ftp = int(round(best_ftp_est))
    else:
        prior_ftps = df[df["date"] <= end_dt]["effective_ftp"].dropna()
        effective_ftp = int(round(float(prior_ftps.iloc[-1]))) if not prior_ftps.empty else config.DEFAULT_FTP

    recent_peaks = recent_90_df["peak_20m_w"].dropna()
    if not recent_peaks.empty:
        peak_20m = round(float(recent_peaks.max()), 1)
        rolling_90d_ftp = round(0.95 * peak_20m, 1)
    else:
        peak_20m = None
        rolling_90d_ftp = None

    window_365_start = end_dt - timedelta(days=364)
    recent_365_df = df[(df["date"] >= window_365_start) & (df["date"] <= end_dt)]
    valid_hr_series = recent_365_df["Max Heart Rate"].dropna()
    valid_hr_series = valid_hr_series[(valid_hr_series >= 135) & (valid_hr_series <= 208)]
    if not valid_hr_series.empty:
        effective_max_hr = int(round(float(valid_hr_series.max())))
    else:
        eff_hr_series = recent_365_df["effective_max_hr"].dropna()
        eff_hr_series = eff_hr_series[(eff_hr_series >= 135) & (eff_hr_series <= 208)]
        effective_max_hr = (
            int(round(float(eff_hr_series.max())))
            if not eff_hr_series.empty
            else config.DEFAULT_MAX_HR
        )

    latest = history[-1]
    return {
        "as_of_date": end_dt.isoformat(),
        "ctl": latest["ctl"],
        "atl": latest["atl"],
        "tsb": latest["tsb"],
        "effective_ftp": effective_ftp,
        "effective_max_hr": effective_max_hr,
        "peak_power_90d_source": peak_power_source,
        "rolling_90d_ftp": rolling_90d_ftp,
        "rolling_90d_peak_20m_w": peak_20m,
        "configured_ftp": config.DEFAULT_FTP,
        "total_activities": total_activities,
        "history": history,
    }


_VALID_PLAN_STATUSES = {"active", "completed", "archived"}


def _utc_now_iso() -> str:
    """Return the current UTC timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _validate_iso_date(value: str | None, field_name: str) -> str | None:
    """Validate and normalize an optional YYYY-MM-DD date string."""
    if value is None:
        return None
    cleaned = value.strip()
    if not cleaned:
        return None
    try:
        return date.fromisoformat(cleaned).isoformat()
    except ValueError as exc:
        raise ValueError(f"{field_name} must be a valid YYYY-MM-DD date, got {value!r}") from exc


def _row_to_note_dict(row: sqlite3.Row) -> dict[str, Any]:
    """Convert an athlete_notes SQLite row into a JSON-serializable dict."""
    data = dict(row)
    data["is_active"] = bool(data["is_active"])
    return data


def save_training_plan(
    title: str,
    plan_content: str,
    goal: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    replace_active: bool = True,
    db_path: Path | None = None,
) -> dict[str, Any]:
    """Store a new training plan in SQLite, optionally archiving prior active plans."""
    clean_title = title.strip()
    clean_content = plan_content.strip()
    if not clean_title:
        raise ValueError("title cannot be empty.")
    if not clean_content:
        raise ValueError("plan_content cannot be empty.")

    clean_goal = goal.strip() if goal and goal.strip() else None
    clean_start = _validate_iso_date(start_date, "start_date")
    clean_end = _validate_iso_date(end_date, "end_date")
    now = _utc_now_iso()

    init_db(db_path)
    with contextlib.closing(get_connection(db_path)) as conn:
        if replace_active:
            conn.execute(
                "UPDATE training_plans SET status = 'archived', updated_at = ? WHERE status = 'active'",
                (now,),
            )
        cursor = conn.execute(
            """
            INSERT INTO training_plans (
                title, goal, start_date, end_date, plan_content, status, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, 'active', ?, ?)
            """,
            (clean_title, clean_goal, clean_start, clean_end, clean_content, now, now),
        )
        plan_id = cursor.lastrowid
        conn.commit()
        row = conn.execute("SELECT * FROM training_plans WHERE id = ?", (plan_id,)).fetchone()
    return dict(row)


def get_active_training_plan(db_path: Path | None = None) -> dict[str, Any] | None:
    """Return the most recently updated active training plan, if one exists."""
    init_db(db_path)
    with contextlib.closing(get_connection(db_path)) as conn:
        row = conn.execute(
            """
            SELECT *
            FROM training_plans
            WHERE status = 'active'
            ORDER BY updated_at DESC, id DESC
            LIMIT 1
            """
        ).fetchone()
    return dict(row) if row is not None else None


def get_training_plan_by_id(plan_id: int, db_path: Path | None = None) -> dict[str, Any] | None:
    """Return a single training plan by ID, or None if not found."""
    init_db(db_path)
    with contextlib.closing(get_connection(db_path)) as conn:
        row = conn.execute("SELECT * FROM training_plans WHERE id = ?", (int(plan_id),)).fetchone()
    return dict(row) if row is not None else None


def list_training_plans(
    include_archived: bool = False,
    limit: int = 10,
    db_path: Path | None = None,
) -> list[dict[str, Any]]:
    """List stored training plans ordered by most recently updated."""
    init_db(db_path)
    clamped_limit = max(1, min(int(limit), 100))
    with contextlib.closing(get_connection(db_path)) as conn:
        if include_archived:
            rows = conn.execute(
                "SELECT * FROM training_plans ORDER BY updated_at DESC, id DESC LIMIT ?",
                (clamped_limit,),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT *
                FROM training_plans
                WHERE status = 'active'
                ORDER BY updated_at DESC, id DESC
                LIMIT ?
                """,
                (clamped_limit,),
            ).fetchall()
    return [dict(r) for r in rows]


def update_training_plan(
    plan_id: int,
    title: str | None = None,
    plan_content: str | None = None,
    goal: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    status: str | None = None,
    db_path: Path | None = None,
) -> dict[str, Any] | None:
    """Update an existing training plan's fields or status."""
    init_db(db_path)
    updates: list[str] = []
    params: list[Any] = []

    if title is not None:
        clean_title = title.strip()
        if not clean_title:
            raise ValueError("title cannot be empty.")
        updates.append("title = ?")
        params.append(clean_title)

    if plan_content is not None:
        clean_content = plan_content.strip()
        if not clean_content:
            raise ValueError("plan_content cannot be empty.")
        updates.append("plan_content = ?")
        params.append(clean_content)

    if goal is not None:
        updates.append("goal = ?")
        params.append(goal.strip() or None)

    if start_date is not None:
        updates.append("start_date = ?")
        params.append(_validate_iso_date(start_date, "start_date"))

    if end_date is not None:
        updates.append("end_date = ?")
        params.append(_validate_iso_date(end_date, "end_date"))

    clean_status: str | None = None
    if status is not None:
        clean_status = status.strip().lower()
        if clean_status not in _VALID_PLAN_STATUSES:
            raise ValueError(
                f"status must be one of {sorted(_VALID_PLAN_STATUSES)}, got {status!r}"
            )
        updates.append("status = ?")
        params.append(clean_status)

    now = _utc_now_iso()
    with contextlib.closing(get_connection(db_path)) as conn:
        existing = conn.execute(
            "SELECT id FROM training_plans WHERE id = ?", (int(plan_id),)
        ).fetchone()
        if existing is None:
            return None

        if clean_status == "active":
            conn.execute(
                """
                UPDATE training_plans
                SET status = 'archived', updated_at = ?
                WHERE status = 'active' AND id != ?
                """,
                (now, int(plan_id)),
            )

        if updates:
            updates.append("updated_at = ?")
            params.append(now)
            params.append(int(plan_id))
            conn.execute(
                f"UPDATE training_plans SET {', '.join(updates)} WHERE id = ?",
                params,
            )
            conn.commit()

        row = conn.execute("SELECT * FROM training_plans WHERE id = ?", (int(plan_id),)).fetchone()
    return dict(row) if row is not None else None


def save_athlete_note(
    content: str,
    category: str = "general",
    effective_from: str | None = None,
    effective_until: str | None = None,
    db_path: Path | None = None,
) -> dict[str, Any]:
    """Save a persistent athlete constraint, schedule rule, preference, or goal."""
    clean_content = content.strip()
    if not clean_content:
        raise ValueError("content cannot be empty.")

    clean_category = (category or "general").strip().lower() or "general"
    clean_from = _validate_iso_date(effective_from, "effective_from")
    clean_until = _validate_iso_date(effective_until, "effective_until")
    now = _utc_now_iso()

    init_db(db_path)
    with contextlib.closing(get_connection(db_path)) as conn:
        cursor = conn.execute(
            """
            INSERT INTO athlete_notes (
                category, content, effective_from, effective_until, is_active, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, 1, ?, ?)
            """,
            (clean_category, clean_content, clean_from, clean_until, now, now),
        )
        note_id = cursor.lastrowid
        conn.commit()
        row = conn.execute("SELECT * FROM athlete_notes WHERE id = ?", (note_id,)).fetchone()
    return _row_to_note_dict(row)


def get_athlete_notes(
    active_only: bool = True,
    include_expired: bool = False,
    category: str | None = None,
    as_of_date: str | None = None,
    db_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Retrieve stored athlete notes, defaulting to active and non-expired entries."""
    init_db(db_path)
    today_iso = _validate_iso_date(as_of_date, "as_of_date") or date.today().isoformat()

    clauses: list[str] = []
    params: list[Any] = []

    if active_only:
        clauses.append("is_active = 1")
    if not include_expired:
        clauses.append("(effective_until IS NULL OR effective_until >= ?)")
        params.append(today_iso)
    if category is not None and category.strip():
        clauses.append("category = ?")
        params.append(category.strip().lower())

    where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    sql = f"""
        SELECT *
        FROM athlete_notes
        {where_sql}
        ORDER BY category ASC, updated_at DESC, id DESC
    """
    with contextlib.closing(get_connection(db_path)) as conn:
        rows = conn.execute(sql, params).fetchall()
    return [_row_to_note_dict(r) for r in rows]


def update_athlete_note(
    note_id: int,
    content: str | None = None,
    category: str | None = None,
    effective_from: str | None = None,
    effective_until: str | None = None,
    is_active: bool | None = None,
    clear_effective_until: bool = False,
    db_path: Path | None = None,
) -> dict[str, Any] | None:
    """Update or deactivate an existing athlete note by ID."""
    init_db(db_path)
    updates: list[str] = []
    params: list[Any] = []

    if content is not None:
        clean_content = content.strip()
        if not clean_content:
            raise ValueError("content cannot be empty.")
        updates.append("content = ?")
        params.append(clean_content)

    if category is not None:
        clean_category = category.strip().lower()
        if not clean_category:
            raise ValueError("category cannot be empty.")
        updates.append("category = ?")
        params.append(clean_category)

    if effective_from is not None:
        updates.append("effective_from = ?")
        params.append(_validate_iso_date(effective_from, "effective_from"))

    if clear_effective_until:
        updates.append("effective_until = NULL")
    elif effective_until is not None:
        updates.append("effective_until = ?")
        params.append(_validate_iso_date(effective_until, "effective_until"))

    if is_active is not None:
        updates.append("is_active = ?")
        params.append(1 if is_active else 0)

    now = _utc_now_iso()
    with contextlib.closing(get_connection(db_path)) as conn:
        existing = conn.execute(
            "SELECT id FROM athlete_notes WHERE id = ?", (int(note_id),)
        ).fetchone()
        if existing is None:
            return None

        if updates:
            updates.append("updated_at = ?")
            params.append(now)
            params.append(int(note_id))
            conn.execute(
                f"UPDATE athlete_notes SET {', '.join(updates)} WHERE id = ?",
                params,
            )
            conn.commit()

        row = conn.execute("SELECT * FROM athlete_notes WHERE id = ?", (int(note_id),)).fetchone()
    return _row_to_note_dict(row) if row is not None else None


def _row_to_timeline_dict(row: sqlite3.Row) -> dict[str, Any]:
    """Convert an activity_timelines SQLite row into a JSON-serializable dict."""
    data = dict(row)
    raw_segments = data.pop("segments_json", "[]")
    raw_highlights = data.pop("highlights_json", None)
    try:
        data["segments"] = json.loads(raw_segments) if raw_segments else []
    except (TypeError, ValueError):
        data["segments"] = []
    try:
        data["highlights"] = json.loads(raw_highlights) if raw_highlights else {}
    except (TypeError, ValueError):
        data["highlights"] = {}
    return data


def upsert_activity_timelines(
    timelines: list[dict[str, Any]],
    db_path: Path | None = None,
) -> int:
    """
    Insert or update compiled activity timeline summaries in activity_timelines.
    Each item should contain: activity_id, summary_text, segments (list),
    and optional source, duration_sec, highlights (dict).
    """
    if not timelines:
        return 0

    init_db(db_path)
    now = _utc_now_iso()
    sql = """
        INSERT INTO activity_timelines (
            activity_id, source, duration_sec, segment_count,
            summary_text, segments_json, highlights_json, created_at, updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(activity_id) DO UPDATE SET
            source = excluded.source,
            duration_sec = excluded.duration_sec,
            segment_count = excluded.segment_count,
            summary_text = excluded.summary_text,
            segments_json = excluded.segments_json,
            highlights_json = excluded.highlights_json,
            updated_at = excluded.updated_at
    """

    rows_to_write: list[tuple[Any, ...]] = []
    for item in timelines:
        act_id = _clean_value(item.get("activity_id"), "Activity ID")
        summary_text = (item.get("summary_text") or "").strip()
        if act_id is None or not summary_text:
            continue
        segments = item.get("segments") or []
        highlights = item.get("highlights") or {}
        duration_sec = item.get("duration_sec")
        if duration_sec is not None:
            try:
                duration_sec = int(round(float(duration_sec)))
            except (TypeError, ValueError):
                duration_sec = None
        rows_to_write.append(
            (
                int(act_id),
                (item.get("source") or "intervals").strip() or "intervals",
                duration_sec,
                len(segments),
                summary_text,
                json.dumps(segments, ensure_ascii=False),
                json.dumps(highlights, ensure_ascii=False),
                now,
                now,
            )
        )

    if not rows_to_write:
        return 0

    with contextlib.closing(get_connection(db_path)) as conn:
        conn.executemany(sql, rows_to_write)
        conn.commit()
    return len(rows_to_write)


def upsert_activity_timeline(
    activity_id: int,
    timeline: dict[str, Any],
    source: str = "intervals",
    db_path: Path | None = None,
) -> bool:
    """Insert or update a single compiled activity timeline."""
    payload = dict(timeline)
    payload["activity_id"] = activity_id
    payload.setdefault("source", source)
    return upsert_activity_timelines([payload], db_path=db_path) > 0


def get_activity_timeline(
    activity_id: int,
    db_path: Path | None = None,
) -> dict[str, Any] | None:
    """Retrieve the compiled timeline (summary, segments, highlights) for an activity."""
    init_db(db_path)
    with contextlib.closing(get_connection(db_path)) as conn:
        row = conn.execute(
            "SELECT * FROM activity_timelines WHERE activity_id = ?",
            (int(activity_id),),
        ).fetchone()
    return _row_to_timeline_dict(row) if row is not None else None


def get_activity_timelines_map(
    activity_ids: list[int],
    db_path: Path | None = None,
) -> dict[int, dict[str, Any]]:
    """Retrieve a mapping of activity_id -> parsed timeline dict for the given IDs."""
    if not activity_ids:
        return {}
    init_db(db_path)
    clean_ids = [int(aid) for aid in activity_ids]
    placeholders = ", ".join("?" for _ in clean_ids)
    with contextlib.closing(get_connection(db_path)) as conn:
        rows = conn.execute(
            f"SELECT * FROM activity_timelines WHERE activity_id IN ({placeholders})",
            clean_ids,
        ).fetchall()
    return {int(row["activity_id"]): _row_to_timeline_dict(row) for row in rows}


def get_existing_timeline_ids(db_path: Path | None = None) -> set[int]:
    """Return the set of activity_ids that already have a compiled timeline stored."""
    init_db(db_path)
    with contextlib.closing(get_connection(db_path)) as conn:
        rows = conn.execute("SELECT activity_id FROM activity_timelines").fetchall()
    return {int(row[0]) for row in rows}
