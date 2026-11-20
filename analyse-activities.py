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


def _format_mmss(seconds: float) -> str:
    """Format elapsed seconds as MM:SS or H:MM:SS."""
    total = max(0, int(round(seconds)))
    hrs, rem = divmod(total, 3600)
    mins, secs = divmod(rem, 60)
    if hrs > 0:
        return f"{hrs}:{mins:02d}:{secs:02d}"
    return f"{mins:02d}:{secs:02d}"


def _mask_to_spans(mask: np.ndarray, max_gap: int = 15) -> list[tuple[int, int]]:
    """
    Convert a boolean 1D array into inclusive (start, end) spans,
    bridging False gaps of length <= max_gap.
    """
    n = len(mask)
    if n == 0 or not np.any(mask):
        return []

    bridged = mask.copy()
    if max_gap > 0:
        idx = np.flatnonzero(bridged)
        if len(idx) > 1:
            gaps = np.diff(idx) - 1
            for pos in np.flatnonzero((gaps > 0) & (gaps <= max_gap)):
                bridged[idx[pos] + 1 : idx[pos + 1]] = True

    padded = np.concatenate(([False], bridged, [False]))
    changes = np.diff(padded.astype(int))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1) - 1
    return list(zip(starts.tolist(), ends.tolist()))


def _power_zone_label(pwr_pct: float) -> str:
    if pwr_pct < 0.55:
        return "Recovery"
    if pwr_pct < 0.75:
        return "Endurance"
    if pwr_pct < 0.90:
        return "Tempo"
    if pwr_pct < 1.05:
        return "Threshold"
    if pwr_pct < 1.20:
        return "VO2max"
    return "Anaerobic"


def _hr_zone_label(hr_pct: float) -> str:
    if hr_pct < 0.60:
        return "Z1 Recovery"
    if hr_pct < 0.70:
        return "Z2 Aerobic"
    if hr_pct < 0.80:
        return "Z3 Tempo"
    if hr_pct < 0.90:
        return "Z4 Threshold"
    return "Z5 Max"


def compile_activity_timeline(
    df: pd.DataFrame,
    effective_ftp: int | None = None,
    effective_max_hr: int | None = None,
) -> dict | None:
    """
    Compile a compact chronological race/ride progression timeline from 1Hz stream data.
    Detects climbs, descents, hard efforts/attacks, start launches, final kicks, and
    steady pacing blocks, returning structured segments, key highlights, and a concise
    text summary suitable for AI coaching context.
    """
    if df is None or df.empty or len(df) < 60:
        return None

    df = df.reset_index(drop=True)
    ftp = int(effective_ftp) if effective_ftp and effective_ftp > 0 else DEFAULT_FTP
    max_hr = int(effective_max_hr) if effective_max_hr and effective_max_hr > 0 else DEFAULT_MAX_HR
    n = len(df)

    pwr = pd.to_numeric(df["power"], errors="coerce") if "power" in df.columns else pd.Series(np.nan, index=df.index)
    hr = pd.to_numeric(df["hr"], errors="coerce") if "hr" in df.columns else pd.Series(np.nan, index=df.index)
    cad = pd.to_numeric(df["cadence"], errors="coerce") if "cadence" in df.columns else pd.Series(np.nan, index=df.index)
    spd = pd.to_numeric(df["speed"], errors="coerce") if "speed" in df.columns else pd.Series(np.nan, index=df.index)
    alt = pd.to_numeric(df["altitude"], errors="coerce") if "altitude" in df.columns else pd.Series(np.nan, index=df.index)

    has_power = pwr.notna().sum() >= 30
    has_hr = hr.notna().sum() >= 30
    has_alt = alt.notna().sum() >= 30
    has_spd = spd.notna().sum() >= 30

    if not any([has_power, has_hr, has_alt]):
        return None

    # Elapsed time vector (seconds from start)
    elapsed = np.arange(n, dtype=float)
    if "time" in df.columns:
        t_col = pd.to_numeric(df["time"], errors="coerce")
        if t_col.notna().sum() == n and t_col.is_monotonic_increasing:
            elapsed = (t_col - t_col.iloc[0]).to_numpy(dtype=float)
    elif "timestamp" in df.columns:
        ts_col = pd.to_datetime(df["timestamp"], errors="coerce")
        if ts_col.notna().sum() == n and ts_col.is_monotonic_increasing:
            elapsed = (ts_col - ts_col.iloc[0]).dt.total_seconds().to_numpy(dtype=float)

    total_duration_sec = max(int(round(elapsed[-1] - elapsed[0])), n)

    # Distance vector (meters)
    dist_m = np.full(n, np.nan, dtype=float)
    if "distance" in df.columns:
        d_col = pd.to_numeric(df["distance"], errors="coerce")
        if d_col.notna().sum() >= 30:
            d_interp = d_col.interpolate(limit_direction="both")
            dist_m = (d_interp - d_interp.iloc[0]).to_numpy(dtype=float)
    if np.isnan(dist_m).all() and has_spd:
        dt = np.diff(elapsed, prepend=elapsed[0])
        dt = np.clip(dt, 0.0, 5.0)
        dist_m = np.cumsum(spd.fillna(0.0).clip(lower=0.0).to_numpy(dtype=float) * dt)
    has_dist = not np.isnan(dist_m).all() and float(np.nanmax(dist_m)) > 100.0

    # Smoothed altitude and gradient (%)
    grade_15s = np.zeros(n, dtype=float)
    alt_smooth = alt.interpolate(limit_direction="both").rolling(15, center=True, min_periods=1).mean() if has_alt else None
    if has_alt and has_dist and alt_smooth is not None:
        alt_diff = alt_smooth.diff().fillna(0.0)
        dist_step = pd.Series(dist_m).diff().fillna(0.0).clip(lower=0.0)
        elev_win = alt_diff.rolling(15, center=True, min_periods=5).sum()
        dist_win = dist_step.rolling(15, center=True, min_periods=5).sum()
        valid_move = dist_win >= 20.0  # moving at least ~1.3 m/s over 15s
        raw_grade = np.where(valid_move, (elev_win / dist_win.replace(0, np.nan)) * 100.0, 0.0)
        grade_15s = np.clip(np.nan_to_num(raw_grade, nan=0.0), -25.0, 25.0)

    # Smoothed power / HR
    pwr_15s = pwr.rolling(15, center=True, min_periods=1).mean().to_numpy(dtype=float) if has_power else np.full(n, np.nan)
    pwr_5s = pwr.rolling(5, min_periods=1).mean().to_numpy(dtype=float) if has_power else np.full(n, np.nan)
    hr_15s = hr.rolling(15, center=True, min_periods=1).mean().to_numpy(dtype=float) if has_hr else np.full(n, np.nan)

    # Primary tag per second: 'steady', 'climb', 'descent', 'effort', 'sprint'
    tags = np.full(n, "steady", dtype=object)

    # 1. Climbs (grade >= 2.5%, duration >= 45s, net gain >= 15m)
    if has_alt and has_dist and alt_smooth is not None:
        for s, e in _mask_to_spans(grade_15s >= 2.5, max_gap=20):
            dur = e - s + 1
            net_gain = float(alt_smooth.iloc[e] - alt_smooth.iloc[s])
            seg_dist = float(dist_m[e] - dist_m[s])
            avg_g = (net_gain / seg_dist * 100.0) if seg_dist > 50 else 0.0
            if dur >= 45 and (net_gain >= 15.0 or (avg_g >= 3.0 and net_gain >= 10.0)):
                tags[s : e + 1] = "climb"

        # 2. Descents (grade <= -2.5%, duration >= 45s, net loss >= 15m)
        for s, e in _mask_to_spans((grade_15s <= -2.5) & (tags == "steady"), max_gap=20):
            dur = e - s + 1
            net_loss = float(alt_smooth.iloc[s] - alt_smooth.iloc[e])
            if dur >= 45 and net_loss >= 15.0:
                tags[s : e + 1] = "descent"

    # 3. Hard efforts / attacks / sprints (on non-climb segments)
    if has_power:
        # High-power surges (>= 125% FTP for >= 15s)
        surge_mask = (np.nan_to_num(pwr_15s, nan=0.0) >= ftp * 1.25) & (tags != "climb")
        for s, e in _mask_to_spans(surge_mask, max_gap=10):
            if (e - s + 1) >= 15:
                tags[s : e + 1] = "sprint" if (n - 1 - e) <= 90 else "effort"

        # Sustained threshold+/VO2 efforts (>= 105% FTP for >= 30s)
        effort_mask = (np.nan_to_num(pwr_15s, nan=0.0) >= ftp * 1.05) & (tags == "steady")
        for s, e in _mask_to_spans(effort_mask, max_gap=15):
            if (e - s + 1) >= 30:
                tags[s : e + 1] = "effort"

        # Final kick in last 2 minutes if strong finish
        tail_start = max(0, n - 120)
        kick_mask = np.zeros(n, dtype=bool)
        kick_mask[tail_start:] = np.nan_to_num(pwr_5s[tail_start:], nan=0.0) >= ftp * 1.20
        for s, e in _mask_to_spans(kick_mask, max_gap=8):
            if (e - s + 1) >= 10 and (n - 1 - e) <= 30:
                tags[s : e + 1] = "sprint"
    elif has_hr:
        hr_effort_mask = (np.nan_to_num(hr_15s, nan=0.0) >= max_hr * 0.85) & (tags == "steady")
        for s, e in _mask_to_spans(hr_effort_mask, max_gap=15):
            if (e - s + 1) >= 45:
                tags[s : e + 1] = "effort"

    # Convert tags array into contiguous (start, end, tag) runs
    runs: list[list] = []
    cur_start = 0
    cur_tag = tags[0]
    for i in range(1, n):
        if tags[i] != cur_tag:
            runs.append([cur_start, i - 1, cur_tag])
            cur_start = i
            cur_tag = tags[i]
    runs.append([cur_start, n - 1, cur_tag])

    # Merge tiny orphan runs (< 30s) unless sprint or strong effort
    cleaned_runs: list[list] = []
    for run in runs:
        s, e, tag = run
        dur = e - s + 1
        is_tiny = dur < 30 and tag not in ("sprint",)
        if is_tiny and cleaned_runs:
            cleaned_runs[-1][1] = e
        else:
            cleaned_runs.append(run)
    # If the very first run was tiny and got kept, merge forward
    if len(cleaned_runs) > 1 and (cleaned_runs[0][1] - cleaned_runs[0][0] + 1) < 25 and cleaned_runs[0][2] == "steady":
        cleaned_runs[1][0] = cleaned_runs[0][0]
        cleaned_runs.pop(0)

    # Merge consecutive runs of identical tag
    merged_runs: list[list] = []
    for run in cleaned_runs:
        if merged_runs and merged_runs[-1][2] == run[2]:
            merged_runs[-1][1] = run[1]
        else:
            merged_runs.append(run)

    # Split long steady stretches into adaptive pacing blocks so drift/fade is visible
    if total_duration_sec <= 5400:
        max_steady_sec = 600   # 10-min blocks for rides <= 90 min
    elif total_duration_sec <= 10800:
        max_steady_sec = 900   # 15-min blocks for rides <= 3h
    else:
        max_steady_sec = 1200  # 20-min blocks for longer rides

    final_runs: list[list] = []
    for s, e, tag in merged_runs:
        dur = e - s + 1
        if tag == "steady" and dur > int(max_steady_sec * 1.4):
            n_chunks = max(2, int(round(dur / max_steady_sec)))
            edges = np.linspace(s, e + 1, n_chunks + 1, dtype=int)
            for k in range(n_chunks):
                cs, ce = int(edges[k]), int(edges[k + 1]) - 1
                if ce >= cs:
                    final_runs.append([cs, ce, tag])
        else:
            final_runs.append([s, e, tag])

    # Cap total segments at 24 to keep context compact (e.g. for 40x 30/30 interval sessions)
    max_segments = 24
    while len(final_runs) > max_segments:
        # Find shortest adjacent pair that isn't a climb or sprint
        best_idx = None
        best_len = float("inf")
        for idx in range(len(final_runs) - 1):
            t1, t2 = final_runs[idx][2], final_runs[idx + 1][2]
            if "climb" in (t1, t2) or "sprint" in (t1, t2):
                continue
            combined_len = final_runs[idx + 1][1] - final_runs[idx][0] + 1
            if combined_len < best_len:
                best_len = combined_len
                best_idx = idx
        if best_idx is None:
            # Fallback: merge shortest adjacent pair overall
            best_idx = min(
                range(len(final_runs) - 1),
                key=lambda i: final_runs[i + 1][1] - final_runs[i][0],
            )
        r1, r2 = final_runs[best_idx], final_runs[best_idx + 1]
        d1 = r1[1] - r1[0] + 1
        d2 = r2[1] - r2[0] + 1
        dominant_tag = r1[2] if d1 >= d2 else r2[2]
        final_runs[best_idx : best_idx + 2] = [[r1[0], r2[1], dominant_tag]]

    # Build rich per-segment dictionaries
    segments: list[dict] = []
    climb_idx = 0
    for seq, (s, e, tag) in enumerate(final_runs, start=1):
        seg_start_sec = int(round(elapsed[s]))
        seg_end_sec = int(round(elapsed[e]))
        seg_dur = max(1, seg_end_sec - seg_start_sec) if seg_end_sec > seg_start_sec else (e - s + 1)

        seg_pwr = pwr.iloc[s : e + 1].dropna()
        seg_hr = hr.iloc[s : e + 1].dropna()
        seg_cad = cad.iloc[s : e + 1].dropna()
        seg_spd = spd.iloc[s : e + 1].dropna()

        avg_pwr = round(float(seg_pwr.mean()), 1) if len(seg_pwr) > 0 else None
        peak_5s_w = round(float(np.nanmax(pwr_5s[s : e + 1])), 1) if len(seg_pwr) > 0 else None
        pwr_pct = round((avg_pwr / ftp) * 100.0, 1) if (avg_pwr is not None and ftp > 0) else None
        avg_hr_val = int(round(float(seg_hr.mean()))) if len(seg_hr) > 0 else None
        max_hr_val = int(round(float(seg_hr.max()))) if len(seg_hr) > 0 else None
        avg_cad_val = int(round(float(seg_cad.mean()))) if len(seg_cad) > 0 else None
        avg_spd_kph = round(float(seg_spd.mean()) * 3.6, 1) if len(seg_spd) > 0 else None

        start_km = round(float(dist_m[s]) / 1000.0, 2) if has_dist else None
        end_km = round(float(dist_m[e]) / 1000.0, 2) if has_dist else None
        seg_dist_km = round(max(0.0, float(dist_m[e] - dist_m[s])) / 1000.0, 2) if has_dist else None

        elev_net_m = None
        avg_grade_pct = None
        if has_alt and alt_smooth is not None:
            elev_net_m = round(float(alt_smooth.iloc[e] - alt_smooth.iloc[s]), 1)
            seg_dist_m = float(dist_m[e] - dist_m[s]) if has_dist else 0.0
            if seg_dist_m >= 50.0:
                avg_grade_pct = round((elev_net_m / seg_dist_m) * 100.0, 1)

        # Human-readable label
        if avg_pwr is not None and ftp > 0:
            zone_lbl = _power_zone_label(avg_pwr / ftp)
        elif avg_hr_val is not None and max_hr > 0:
            zone_lbl = _hr_zone_label(avg_hr_val / max_hr)
        else:
            zone_lbl = "Ride"

        if tag == "climb":
            climb_idx += 1
            label = f"Climb #{climb_idx} ({zone_lbl})"
        elif tag == "descent":
            label = f"Descent ({zone_lbl})"
        elif tag == "sprint":
            label = "Final Kick / Sprint" if (n - 1 - e) <= 60 else "Surge / Sprint"
        elif tag == "effort":
            if seq == 1 and seg_start_sec <= 30 and seg_dur <= 210:
                label = f"Start / Launch ({zone_lbl})"
            else:
                label = f"Hard Effort ({zone_lbl})"
        else:
            if seq == 1 and seg_dur <= 180 and pwr_pct is not None and pwr_pct >= 100.0:
                label = f"Start / Launch ({zone_lbl})"
            else:
                label = f"Steady ({zone_lbl})"

        segments.append({
            "seq": seq,
            "start_sec": seg_start_sec,
            "end_sec": seg_end_sec,
            "duration_sec": seg_dur,
            "time_range": f"{_format_mmss(seg_start_sec)}-{_format_mmss(seg_end_sec)}",
            "start_km": start_km,
            "end_km": end_km,
            "distance_km": seg_dist_km,
            "segment_type": tag,
            "label": label,
            "avg_power_w": avg_pwr,
            "peak_5s_power_w": peak_5s_w,
            "power_pct_ftp": pwr_pct,
            "avg_hr": avg_hr_val,
            "max_hr": max_hr_val,
            "avg_cadence": avg_cad_val,
            "elev_change_m": elev_net_m,
            "avg_grade_pct": avg_grade_pct,
            "avg_speed_kph": avg_spd_kph,
        })

    # Key highlights (peak effort timestamps + pacing split)
    highlights: dict = {
        "effective_ftp": ftp,
        "effective_max_hr": max_hr,
        "climbs_count": sum(1 for seg in segments if seg["segment_type"] == "climb"),
        "hard_efforts_count": sum(1 for seg in segments if seg["segment_type"] in ("effort", "sprint")),
    }

    if has_power:
        pwr_filled = pwr.ffill().bfill()
        for key, win in [("peak_15s", 15), ("peak_1m", 60), ("peak_5m", 300)]:
            if len(pwr_filled) >= win:
                roll = pwr_filled.rolling(win, min_periods=win).mean()
                best_end = int(roll.idxmax())
                best_val = float(roll.iloc[best_end])
                best_start = max(0, best_end - win + 1)
                highlights[key] = {
                    "watts": round(best_val, 1),
                    "pct_ftp": round((best_val / ftp) * 100.0, 1) if ftp > 0 else None,
                    "start_sec": int(round(elapsed[best_start])),
                    "time_at": _format_mmss(elapsed[best_start]),
                }

    # First-half vs second-half pacing split
    mid = n // 2
    if mid >= 30:
        split: dict = {}
        if has_power:
            p1 = pwr.iloc[:mid].dropna()
            p2 = pwr.iloc[mid:].dropna()
            if len(p1) > 10 and len(p2) > 10:
                m1, m2 = float(p1.mean()), float(p2.mean())
                split["first_half_w"] = round(m1, 1)
                split["second_half_w"] = round(m2, 1)
                if m1 > 0:
                    split["power_change_pct"] = round(((m2 - m1) / m1) * 100.0, 1)
        if has_hr:
            h1 = hr.iloc[:mid].dropna()
            h2 = hr.iloc[mid:].dropna()
            if len(h1) > 10 and len(h2) > 10:
                split["first_half_hr"] = int(round(float(h1.mean())))
                split["second_half_hr"] = int(round(float(h2.mean())))
        if split:
            highlights["pacing_split"] = split

    # Render compact human/LLM-readable text summary
    dur_min = round(total_duration_sec / 60.0, 1)
    header_parts = [f"{dur_min:.0f}m" if dur_min >= 10 else f"{dur_min:.1f}m"]
    if has_dist:
        header_parts.append(f"{float(dist_m[-1]) / 1000.0:.1f} km")
    header_parts.append(f"FTP {ftp}W")
    lines = [f"Timeline ({', '.join(header_parts)}):"]

    for seg in segments:
        parts = [f"{seg['time_range']}", seg["label"]]
        if seg["avg_power_w"] is not None:
            pwr_str = f"{seg['avg_power_w']:.0f}W ({seg['power_pct_ftp']:.0f}% FTP)"
            if seg["segment_type"] == "sprint" and seg["peak_5s_power_w"]:
                pwr_str += f", pk5s {seg['peak_5s_power_w']:.0f}W"
            parts.append(pwr_str)
        if seg["avg_hr"] is not None:
            parts.append(f"HR {seg['avg_hr']}/{seg['max_hr']} bpm")
        if seg["elev_change_m"] is not None and seg["avg_grade_pct"] is not None:
            if seg["segment_type"] in ("climb", "descent") or abs(seg["elev_change_m"]) >= 15:
                sign = "+" if seg["elev_change_m"] >= 0 else ""
                parts.append(f"{sign}{seg['elev_change_m']:.0f}m @ {seg['avg_grade_pct']:+.1f}%")
        lines.append("- " + " | ".join(parts))

    moment_bits: list[str] = []
    for label, key in [("Peak 15s", "peak_15s"), ("Peak 1m", "peak_1m"), ("Peak 5m", "peak_5m")]:
        pk = highlights.get(key)
        if pk:
            moment_bits.append(f"{label} {pk['watts']:.0f}W @ {pk['time_at']}")
    sp = highlights.get("pacing_split")
    if sp:
        h1_str = []
        h2_str = []
        if "first_half_w" in sp:
            h1_str.append(f"{sp['first_half_w']:.0f}W")
            h2_str.append(f"{sp['second_half_w']:.0f}W")
        if "first_half_hr" in sp:
            h1_str.append(f"{sp['first_half_hr']}bpm")
            h2_str.append(f"{sp['second_half_hr']}bpm")
        if h1_str and h2_str:
            moment_bits.append(f"Halves {'/'.join(h1_str)} -> {'/'.join(h2_str)}")
    if moment_bits:
        lines.append("Key Moments: " + "; ".join(moment_bits))

    return {
        "duration_sec": total_duration_sec,
        "summary_text": "\n".join(lines),
        "segments": segments,
        "highlights": highlights,
    }


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
                            "distance": frame.get_value("distance", fallback=np.nan),
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
        "time": extract_col("time"),
        "distance": extract_col("distance"),
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
    new_timelines = []
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
                "keys": "time,distance,heartrate,watts,cadence,velocity_smooth,altitude",
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
        timeline = compile_activity_timeline(
            stream_df,
            effective_ftp=eff_ftp,
            effective_max_hr=eff_max_hr,
        )
        if timeline is not None:
            timeline["activity_id"] = act_id
            timeline["source"] = "strava"
            new_timelines.append(timeline)

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
    timelines_saved = app_db.upsert_activity_timelines(new_timelines)
    print(
        f"Appended {len(new_df)} new activities to {OUTPUT_CSV} "
        f"(Total rows: {len(combined_df)}, DB inserted: {db_inserted}, timelines saved: {timelines_saved})."
    )
    return {
        "source": "strava",
        "fetched": len(new_df),
        "db_inserted": db_inserted,
        "timelines_saved": timelines_saved,
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
        "time": extract_col("time"),
        "distance": extract_col("distance"),
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


def _fetch_intervals_stream(client: IntervalsAPIClient, raw_id: str) -> tuple[pd.DataFrame, int]:
    """Download processed/original FIT file or fallback JSON streams for an Intervals.icu activity."""
    stream_df = pd.DataFrame()
    lap_count = 0

    for file_endpoint in (f"/activity/{raw_id}/fit-file", f"/activity/{raw_id}/file"):
        try:
            raw_bytes = client.get_bytes(file_endpoint)
            if raw_bytes:
                stream_df, lap_count = parse_fit_bytes(raw_bytes, source_name=f"intervals_{raw_id}")
                if not stream_df.empty:
                    return stream_df, lap_count
        except Exception as exc:
            print(f"Warning: Could not download {file_endpoint}: {exc}")

    try:
        streams_payload = client.get_json(
            f"/activity/{raw_id}/streams.json",
            params={"types": "time,distance,heartrate,watts,cadence,velocity_smooth,altitude"},
        )
        stream_df = intervals_streams_to_dataframe(streams_payload)
    except Exception as exc:
        print(f"Warning: Could not fetch streams for Intervals.icu activity {raw_id}: {exc}")

    return stream_df, lap_count


def _find_matching_existing_id(
    candidate_ts: pd.Timestamp | None,
    existing_ts_pairs: list[tuple[int, pd.Timestamp]],
    tolerance_sec: int = 90,
) -> int | None:
    """Return the Activity ID of an existing row whose timestamp falls within tolerance_sec."""
    if candidate_ts is None or pd.isna(candidate_ts):
        return None
    for act_id, existing_ts in existing_ts_pairs:
        if abs((candidate_ts - existing_ts).total_seconds()) <= tolerance_sec:
            return act_id
    return None


def sync_recent_from_intervals(
    limit: int = 30,
    after_days: int | None = None,
    backfill_timelines: bool = False,
) -> dict:
    """
    Fetches recent activities from Intervals.icu, downloads processed/original .fit files
    (falling back to streams if needed), enriches them via the shared metric engine,
    compiles a chronological timeline summary for each activity, appends new rows to
    OUTPUT_CSV, and inserts activities + timelines into SQLite.
    Deduplicates against both Activity IDs and start timestamps (+/-90s) so historical
    Strava rides are never duplicated.
    Set backfill_timelines=True to also compile timelines for already-synced Intervals.icu
    activities in the lookback window that do not yet have a timeline stored.
    """
    api_key = INTERVALS_API_KEY or os.getenv("INTERVALS_API_KEY", "")
    athlete_id = INTERVALS_ATHLETE_ID or os.getenv("INTERVALS_ATHLETE_ID", "0")
    client = IntervalsAPIClient(api_key=api_key, athlete_id=athlete_id)

    existing_df = pd.DataFrame()
    existing_ids: set[int] = set()
    existing_timestamps: list[pd.Timestamp] = []
    existing_ts_pairs: list[tuple[int, pd.Timestamp]] = []
    existing_phys_by_id: dict[int, tuple[int | None, int | None]] = {}
    if OUTPUT_CSV.exists():
        existing_df = pd.read_csv(OUTPUT_CSV)
        if "Activity ID" in existing_df.columns:
            existing_ids = set(pd.to_numeric(existing_df["Activity ID"], errors="coerce").dropna().astype(int))
            for _, erow in existing_df.iterrows():
                eid = _valid_num(erow.get("Activity ID"))
                if eid is None:
                    continue
                eid_int = int(eid)
                e_ftp = _valid_num(erow.get("effective_ftp"))
                e_hr = _valid_num(erow.get("effective_max_hr"))
                existing_phys_by_id[eid_int] = (
                    int(round(e_ftp)) if e_ftp and e_ftp > 0 else None,
                    int(round(e_hr)) if e_hr and e_hr > 0 else None,
                )
                ets = _to_naive_ts(erow.get("Activity Date"))
                if ets is not None:
                    existing_timestamps.append(ets)
                    existing_ts_pairs.append((eid_int, ets))
        elif "Activity Date" in existing_df.columns:
            parsed_dates = pd.to_datetime(existing_df["Activity Date"], format="mixed", errors="coerce").dropna()
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
        source = str(act.get("source", "")).upper()
        note = str(act.get("_note", "")).upper()
        has_metrics = bool(act.get("type") or act.get("moving_time") or act.get("elapsed_time") or act.get("distance"))
        # Intervals.icu returns blank stub objects (often still containing start_date_local and _note)
        # for Strava-sourced activities because Strava forbids API re-export.
        if "STRAVA" in note or (source == "STRAVA" and not has_metrics):
            skipped_strava_stubs += 1
            continue
        start_iso = act.get("start_date") or act.get("start_date_local")
        if not start_iso or not has_metrics:
            continue
        candidates.append(act)

    # Sort newest-first to respect `limit`, then filter out duplicates
    candidates.sort(key=lambda x: x.get("start_date") or x.get("start_date_local") or "", reverse=True)
    candidates = candidates[: max(1, min(int(limit), 200))]

    existing_timeline_ids = app_db.get_existing_timeline_ids() if backfill_timelines else set()
    new_summaries = []
    backfill_targets: list[tuple[int, dict]] = []
    for act in candidates:
        int_id = intervals_id_to_int(act["id"])
        if int_id in existing_ids:
            if backfill_timelines and int_id not in existing_timeline_ids:
                backfill_targets.append((int_id, act))
            continue
        start_iso = act.get("start_date") or act.get("start_date_local")
        act_ts = _to_naive_ts(start_iso)
        matched_id = _find_matching_existing_id(act_ts, existing_ts_pairs, tolerance_sec=90)
        if matched_id is not None or _is_duplicate_timestamp(act_ts, existing_timestamps, tolerance_sec=90):
            if backfill_timelines and matched_id is not None and matched_id not in existing_timeline_ids:
                backfill_targets.append((matched_id, act))
            continue
        new_summaries.append(act)

    backfilled_timelines: list[dict] = []
    if backfill_targets:
        print(f"Backfilling timelines for {len(backfill_targets)} existing Intervals.icu activities...")
        for b_idx, (target_id, summary) in enumerate(backfill_targets, start=1):
            raw_id = str(summary["id"])
            act_name = summary.get("name") or "Unnamed Activity"
            print(f"[Backfill {b_idx}/{len(backfill_targets)}] Compiling timeline for {raw_id} (DB ID {target_id}): {act_name}...")
            stream_df, _ = _fetch_intervals_stream(client, raw_id)
            eff_ftp, eff_max_hr = existing_phys_by_id.get(target_id, (None, None))
            timeline = compile_activity_timeline(
                stream_df,
                effective_ftp=eff_ftp,
                effective_max_hr=eff_max_hr,
            )
            if timeline is not None:
                timeline["activity_id"] = target_id
                timeline["source"] = "intervals"
                backfilled_timelines.append(timeline)

    if not new_summaries:
        timelines_saved = app_db.upsert_activity_timelines(backfilled_timelines)
        print(
            f"All recent Intervals.icu activities are already present in the dataset "
            f"(skipped_strava_stubs={skipped_strava_stubs}, timelines_saved={timelines_saved})."
        )
        result = {
            "source": "intervals",
            "fetched": 0,
            "db_inserted": 0,
            "timelines_saved": timelines_saved,
            "total_csv_rows": len(existing_df),
            "skipped_strava_stubs": skipped_strava_stubs,
        }
        if skipped_strava_stubs > 0:
            result["note"] = (
                f"Skipped {skipped_strava_stubs} Strava-sourced activity stub(s) that Intervals.icu "
                "cannot export via API due to Strava restrictions. Connect Garmin, Zwift, Wahoo, etc. "
                "directly to Intervals.icu so full activity data is available."
            )
        return result

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
    new_timelines = list(backfilled_timelines)
    for idx, summary in enumerate(new_summaries, start=1):
        raw_id = str(summary["id"])
        int_id = intervals_id_to_int(raw_id)
        act_name = summary.get("name") or "Unnamed Activity"
        print(f"[{idx}/{len(new_summaries)}] Fetching & enriching Intervals.icu Activity {raw_id}: {act_name}...")

        stream_df, lap_count = _fetch_intervals_stream(client, raw_id)

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
        timeline = compile_activity_timeline(
            stream_df,
            effective_ftp=eff_ftp,
            effective_max_hr=eff_max_hr,
        )
        if timeline is not None:
            timeline["activity_id"] = int_id
            timeline["source"] = "intervals"
            new_timelines.append(timeline)

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
    timelines_saved = app_db.upsert_activity_timelines(new_timelines)
    print(
        f"Appended {len(new_df)} new Intervals.icu activities to {OUTPUT_CSV} "
        f"(Total rows: {len(combined_df)}, DB inserted: {db_inserted}, timelines saved: {timelines_saved})."
    )
    return {
        "source": "intervals",
        "fetched": len(new_df),
        "db_inserted": db_inserted,
        "timelines_saved": timelines_saved,
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
    parser.add_argument(
        "--backfill-timelines",
        action="store_true",
        help="In intervals mode, also compile timelines for already-synced activities missing one.",
    )
    args = parser.parse_args()

    if args.mode == "bulk":
        run_bulk_export()
    elif args.mode == "intervals":
        sync_recent_from_intervals(
            limit=args.limit,
            after_days=args.after_days,
            backfill_timelines=args.backfill_timelines,
        )
    else:
        sync_recent_from_api(limit=args.limit, after_days=args.after_days)


if __name__ == "__main__":
    main()