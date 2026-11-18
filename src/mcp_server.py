"""FastMCP server exposing Strava cycling metrics, playbook, and live sync tools."""

from __future__ import annotations

import logging
import re
import sqlite3
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import date, timedelta
from typing import Any

import uvicorn
from mcp.server.fastmcp import FastMCP
from starlette.applications import Starlette

from src import config, db, scheduler

logger = logging.getLogger(__name__)

mcp = FastMCP(
    "StravaCyclingCoach",
    instructions=(
        "Whenever the user asks for cycling or Zwift training advice, session recommendations, "
        "or recovery guidance, ALWAYS call get_training_playbook(), get_current_readiness(), "
        "and get_recent_activities(days=10) first. Base all target wattages on the rolling "
        "effective_ftp and heart rate caps on effective_max_hr."
    ),
    host=config.MCP_HOST,
    port=config.MCP_PORT,
)

_MAX_QUERY_ROWS = 200
_READ_ONLY_PREFIX_RE = re.compile(r"^\s*(SELECT|WITH|EXPLAIN)\b", re.IGNORECASE)


def _load_playbook() -> str:
    """Read playbook.md from DATA_DIR or return a helpful fallback message."""
    if config.PLAYBOOK_PATH.exists():
        return config.PLAYBOOK_PATH.read_text(encoding="utf-8")
    return (
        f"Playbook not found at `{config.PLAYBOOK_PATH}`.\n"
        "Place your personal training correlation analysis in `server/playbook.md` "
        "(mounted at `/app/server/playbook.md` in Docker) to make it available here."
    )

@mcp.prompt()
def daily_coaching_check(
    question: str = "What training should I do today?",
    available_minutes: int = 60,
    how_i_feel: str = "normal",
) -> str:
    """
    Structured coaching prompt that loads playbook rules, readiness (CTL/ATL/TSB/FTP),
    and recent rides before recommending today's session.
    """
    return f"""You are acting as my personal AI cycling coach.

Before answering my question, you MUST call these three MCP tools in order:
1. `get_training_playbook()` — to review my historical responder profile, effective training patterns, and schedule preferences.
2. `get_current_readiness()` — to check my rolling 90-day FTP, 365-day Max HR, CTL (fitness), ATL (fatigue), TSB (form), and 7-day zone totals.
3. `get_recent_activities(days=10)` — to see my recent sessions, intensity factor, aerobic decoupling, and HIIT surges.

Athlete context for today:
- Question / Goal: {question}
- Available Time: {available_minutes} minutes
- Subjective Feeling / Leg Fatigue: {how_i_feel}

Based on the data returned by those tools and my playbook findings:
1. Briefly summarize my current readiness (TSB, recent load, and what training stimulus is currently missing or overloaded).
2. Recommend a specific session (e.g., exact Zwift route/workout type, interval structure, or active recovery spin).
3. Specify exact target wattages and heart rate caps calculated from my current rolling `effective_ftp` and `effective_max_hr`, plus expected TSS.
"""

@mcp.resource("coach://playbook")
def playbook_resource() -> str:
    """Personal cycling training playbook and historical responder rules."""
    return _load_playbook()


@mcp.tool()
def get_training_playbook() -> str:
    """Return the user's personal training playbook (playbook.md) with historical findings."""
    return _load_playbook()


@mcp.tool()
def get_current_readiness() -> dict[str, Any]:
    """
    Compute current CTL (fitness), ATL (fatigue), TSB (form),
    live rolling 90-day effective_ftp, 365-day effective_max_hr,
    the source activity/date for the 90-day peak power, and the last 14 days of load.
    """
    load = db.compute_training_load()
    tsb = load["tsb"]
    if tsb > 5.0:
        status = "Fresh / Race Ready"
    elif tsb >= -10.0:
        status = "Optimal / Neutral"
    elif tsb >= -25.0:
        status = "Productive Training / Fatigued"
    else:
        status = "High Fatigue / Overreaching Risk"

    return {
        "as_of_date": load["as_of_date"],
        "ctl": load["ctl"],
        "atl": load["atl"],
        "tsb": tsb,
        "readiness_status": status,
        "effective_ftp": load["effective_ftp"],
        "effective_max_hr": load["effective_max_hr"],
        "peak_power_90d_source": load["peak_power_90d_source"],
        "rolling_90d_ftp": load["rolling_90d_ftp"],
        "rolling_90d_peak_20m_w": load["rolling_90d_peak_20m_w"],
        "default_ftp": config.DEFAULT_FTP,
        "default_max_hr": config.DEFAULT_MAX_HR,
        "total_activities": load["total_activities"],
        "recent_14d_load": load["history"][-14:],
    }


@mcp.resource("coach://status-summary")
def status_summary_resource() -> str:
    """Live summary of CTL, ATL, TSB, rolling 90-day effective FTP, and 365-day Max HR."""
    r = get_current_readiness()
    peak = r["peak_power_90d_source"]
    if peak:
        peak_line = (
            f"{peak['watts']} W ({peak['metric']}) on {peak['date']} "
            f"— {peak['activity_name']} (ID: {peak['activity_id']})"
        )
    else:
        peak_line = "No 5m/20m power peaks recorded in the last 90 days (using fallback/carry-forward)"

    return (
        f"# Cycling Readiness Summary ({r['as_of_date']})\n\n"
        f"- **Readiness Status:** {r['readiness_status']}\n"
        f"- **Fitness (CTL, 42d):** {r['ctl']}\n"
        f"- **Fatigue (ATL, 7d):** {r['atl']}\n"
        f"- **Form (TSB):** {r['tsb']}\n"
        f"- **Effective FTP (90d rolling):** {r['effective_ftp']} W\n"
        f"- **90d Peak Power Source:** {peak_line}\n"
        f"- **Effective Max HR (365d rolling):** {r['effective_max_hr']} bpm\n"
        f"- **Total Activities Indexed:** {r['total_activities']}\n"
    )


@mcp.tool()
def get_recent_activities(days: int = 14) -> dict[str, Any]:
    """Return enriched activities from the last N days (default: 14, max: 365)."""
    clamped_days = max(1, min(int(days), 365))
    cutoff = (date.today() - timedelta(days=clamped_days)).isoformat()

    db.init_db()
    with db.get_connection() as conn:
        rows = conn.execute(
            """
            SELECT *
            FROM activities
            WHERE activity_date_iso >= ?
            ORDER BY activity_date_iso DESC, "Activity ID" DESC
            """,
            (cutoff,),
        ).fetchall()

    return {
        "days": clamped_days,
        "cutoff_date": cutoff,
        "count": len(rows),
        "activities": [dict(r) for r in rows],
    }


@mcp.tool()
def sync_strava_now(limit: int = 30) -> dict[str, Any]:
    """Trigger an immediate Strava API sync for recent activities (default limit: 30)."""
    clamped_limit = max(1, min(int(limit), 200))
    return scheduler.run_sync(limit=clamped_limit)


@mcp.tool()
def query_training_database(sql_query: str) -> dict[str, Any]:
    """
    Execute a read-only SQLite query (SELECT / WITH / EXPLAIN) against cycling.db.
    Results are capped at 200 rows.
    """
    cleaned = sql_query.strip().rstrip(";").strip()
    if not cleaned:
        return {"error": "SQL query cannot be empty."}
    if ";" in cleaned:
        return {"error": "Multiple SQL statements are not permitted."}
    if not _READ_ONLY_PREFIX_RE.match(cleaned):
        return {"error": "Only read-only SELECT, WITH, or EXPLAIN queries are permitted."}

    db.init_db()
    ro_uri = f"file:{config.DB_PATH.resolve()}?mode=ro"
    try:
        conn = sqlite3.connect(ro_uri, uri=True)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA query_only = ON;")
        try:
            cursor = conn.execute(cleaned)
            rows = cursor.fetchmany(_MAX_QUERY_ROWS + 1)
            truncated = len(rows) > _MAX_QUERY_ROWS
            if truncated:
                rows = rows[:_MAX_QUERY_ROWS]
            return {
                "row_count": len(rows),
                "truncated": truncated,
                "rows": [dict(r) for r in rows],
            }
        finally:
            conn.close()
    except sqlite3.Error as exc:
        return {"error": f"SQLite error: {exc}"}


@asynccontextmanager
async def _lifespan(_app: Starlette) -> AsyncIterator[None]:
    """Manage the background Strava scheduler and FastMCP session manager."""
    sched = scheduler.start_scheduler(run_on_startup=True)
    try:
        async with mcp.session_manager.run():
            yield
    finally:
        scheduler.stop_scheduler(sched)


def create_app() -> Starlette:
    """Build the combined ASGI app exposing Streamable HTTP (/mcp) and SSE (/sse)."""
    http_app = mcp.streamable_http_app()
    sse_app = mcp.sse_app()
    return Starlette(
        routes=[*http_app.routes, *sse_app.routes],
        lifespan=_lifespan,
    )


def main() -> None:
    """Start background Strava scheduler and run FastMCP over Streamable HTTP and SSE."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger.info(
        "Starting StravaCyclingCoach FastMCP server on %s:%s (Streamable HTTP /mcp & SSE /sse)...",
        config.MCP_HOST,
        config.MCP_PORT,
    )
    app = create_app()
    uvicorn.run(app, host=config.MCP_HOST, port=config.MCP_PORT)


if __name__ == "__main__":
    main()
