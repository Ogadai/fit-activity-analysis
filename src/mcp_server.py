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
        "training plans, or recovery guidance, ALWAYS call get_training_playbook(), "
        "get_current_readiness(), get_recent_activities(days=10), get_training_plan(), and "
        "get_athlete_notes() first. Base all target wattages on the rolling effective_ftp and "
        "heart rate caps on effective_max_hr, and respect any active training plan and athlete "
        "notes/constraints (such as available days, injuries, travel, or goals). "
        "Whenever the user shares schedule constraints, availability changes, injuries, upcoming "
        "events, or preferences for future sessions (e.g. 'I can only work out on Mondays and "
        "Tuesdays at the moment'), call save_athlete_note() (or update_athlete_note() if "
        "updating/replacing an existing note) so it is persisted in the database for future chats. "
        "Whenever you create or revise a training plan, call save_training_plan() (or "
        "update_training_plan()) to store it in the database."
    ),
    host=config.MCP_HOST,
    port=config.MCP_PORT,
    stateless_http=True,
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

Before answering my question, you MUST call these MCP tools:
1. `get_training_playbook()` — to review my historical responder profile, effective training patterns, and schedule preferences.
2. `get_current_readiness()` — to check my rolling 90-day FTP, 365-day Max HR, CTL (fitness), ATL (fatigue), TSB (form), and 7-day zone totals.
3. `get_recent_activities(days=10)` — to see my recent sessions, intensity factor, aerobic decoupling, and HIIT surges.
4. `get_training_plan()` — to check my active training plan and where today fits into it.
5. `get_athlete_notes()` — to check any saved schedule constraints, availability, injuries, or preferences.

Athlete context for today:
- Question / Goal: {question}
- Available Time: {available_minutes} minutes
- Subjective Feeling / Leg Fatigue: {how_i_feel}

Based on the data returned by those tools, my active training plan, any saved athlete notes, and my playbook findings:
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
def sync_activities_now(
    limit: int = 30,
    after_days: int | None = None,
    source: str = "auto",
) -> dict[str, Any]:
    """
    Trigger an immediate activity sync from the configured provider.
    source: 'auto' (prefers Intervals.icu if INTERVALS_API_KEY is set, else Strava),
    'intervals', or 'strava'.
    """
    clamped_limit = max(1, min(int(limit), 200))
    return scheduler.run_sync(limit=clamped_limit, after_days=after_days, source=source)


@mcp.tool()
def sync_intervals_now(limit: int = 30, after_days: int | None = None) -> dict[str, Any]:
    """Trigger an immediate Intervals.icu API sync for recent activities (default limit: 30, lookback: 90 days)."""
    clamped_limit = max(1, min(int(limit), 200))
    return scheduler.run_sync(limit=clamped_limit, after_days=after_days, source="intervals")


@mcp.tool()
def sync_strava_now(limit: int = 30) -> dict[str, Any]:
    """Trigger an immediate Strava API sync for recent activities (default limit: 30)."""
    clamped_limit = max(1, min(int(limit), 200))
    return scheduler.run_sync(limit=clamped_limit, source="strava")


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


@mcp.resource("coach://training-plan")
def training_plan_resource() -> str:
    """Currently active cycling training plan stored in SQLite."""
    plan = db.get_active_training_plan()
    if not plan:
        return "No active training plan is currently stored."
    lines = [f"# {plan['title']} (Plan ID: {plan['id']})"]
    if plan.get("goal"):
        lines.append(f"- **Goal:** {plan['goal']}")
    if plan.get("start_date") or plan.get("end_date"):
        lines.append(
            f"- **Dates:** {plan.get('start_date') or 'unspecified'} to {plan.get('end_date') or 'unspecified'}"
        )
    lines.append(f"- **Status:** {plan['status']}")
    lines.append(f"- **Updated:** {plan['updated_at']}")
    lines.append("")
    lines.append(plan["plan_content"])
    return "\n".join(lines)


@mcp.resource("coach://athlete-notes")
def athlete_notes_resource() -> str:
    """Active athlete constraints, availability, injuries, and planning notes."""
    notes = db.get_athlete_notes(active_only=True, include_expired=False)
    if not notes:
        return "No active athlete notes or constraints are currently stored."
    lines = ["# Active Athlete Notes & Constraints", ""]
    for note in notes:
        window = ""
        if note.get("effective_from") or note.get("effective_until"):
            window = f" ({note.get('effective_from') or '...'} -> {note.get('effective_until') or '...'})"
        lines.append(f"- **[ID {note['id']} | {note['category']}{window}]** {note['content']}")
    return "\n".join(lines)


@mcp.tool()
def save_training_plan(
    title: str,
    plan_content: str,
    goal: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    replace_active: bool = True,
) -> dict[str, Any]:
    """
    Save a training plan to SQLite for reference in future chats.
    By default, any previously active plan is archived (replace_active=True).
    Dates must be YYYY-MM-DD if provided.
    """
    try:
        plan = db.save_training_plan(
            title=title,
            plan_content=plan_content,
            goal=goal,
            start_date=start_date,
            end_date=end_date,
            replace_active=replace_active,
        )
        return {"saved": True, "plan": plan}
    except ValueError as exc:
        return {"error": str(exc)}


@mcp.tool()
def get_training_plan(
    include_archived: bool = False,
    plan_id: int | None = None,
) -> dict[str, Any]:
    """
    Retrieve the currently active training plan (or a specific plan by plan_id).
    Set include_archived=True to also list past/archived plans.
    """
    if plan_id is not None:
        plan = db.get_training_plan_by_id(plan_id)
        if plan is None:
            return {"found": False, "plan": None, "error": f"Training plan {plan_id} not found."}
        return {"found": True, "plan": plan}

    active_plan = db.get_active_training_plan()
    result: dict[str, Any] = {
        "has_active_plan": active_plan is not None,
        "active_plan": active_plan,
    }
    if include_archived:
        result["plans"] = db.list_training_plans(include_archived=True, limit=25)
    return result


@mcp.tool()
def update_training_plan(
    plan_id: int,
    title: str | None = None,
    plan_content: str | None = None,
    goal: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    status: str | None = None,
) -> dict[str, Any]:
    """
    Update an existing training plan's title, content, goal, dates, or status
    ('active', 'completed', or 'archived').
    """
    try:
        updated = db.update_training_plan(
            plan_id=plan_id,
            title=title,
            plan_content=plan_content,
            goal=goal,
            start_date=start_date,
            end_date=end_date,
            status=status,
        )
    except ValueError as exc:
        return {"error": str(exc)}

    if updated is None:
        return {"updated": False, "error": f"Training plan {plan_id} not found."}
    return {"updated": True, "plan": updated}


@mcp.tool()
def save_athlete_note(
    content: str,
    category: str = "general",
    effective_from: str | None = None,
    effective_until: str | None = None,
) -> dict[str, Any]:
    """
    Save persistent information about the athlete for future planning and recommendations
    (e.g. 'I can only work out on Mondays and Tuesdays at the moment').
    Suggested categories: 'schedule', 'availability', 'injury', 'goal', 'equipment', 'preference', 'general'.
    Optional effective_from / effective_until dates must be YYYY-MM-DD.
    """
    try:
        note = db.save_athlete_note(
            content=content,
            category=category,
            effective_from=effective_from,
            effective_until=effective_until,
        )
        return {"saved": True, "note": note}
    except ValueError as exc:
        return {"error": str(exc)}


@mcp.tool()
def get_athlete_notes(
    active_only: bool = True,
    include_expired: bool = False,
    category: str | None = None,
) -> dict[str, Any]:
    """
    Retrieve saved athlete notes, schedule constraints, injuries, and preferences.
    By default returns only active, non-expired notes.
    """
    notes = db.get_athlete_notes(
        active_only=active_only,
        include_expired=include_expired,
        category=category,
    )
    return {
        "count": len(notes),
        "notes": notes,
    }


@mcp.tool()
def update_athlete_note(
    note_id: int,
    content: str | None = None,
    category: str | None = None,
    effective_from: str | None = None,
    effective_until: str | None = None,
    is_active: bool | None = None,
    clear_effective_until: bool = False,
) -> dict[str, Any]:
    """
    Update an existing athlete note by note_id, or deactivate it (is_active=False)
    when a constraint no longer applies.
    """
    try:
        updated = db.update_athlete_note(
            note_id=note_id,
            content=content,
            category=category,
            effective_from=effective_from,
            effective_until=effective_until,
            is_active=is_active,
            clear_effective_until=clear_effective_until,
        )
    except ValueError as exc:
        return {"error": str(exc)}

    if updated is None:
        return {"updated": False, "error": f"Athlete note {note_id} not found."}
    return {"updated": True, "note": updated}


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
