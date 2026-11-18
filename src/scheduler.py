"""Background scheduler for daily and startup Strava activity ingestion."""

from __future__ import annotations

import importlib.util
import logging
from datetime import datetime
from pathlib import Path
from types import ModuleType
from typing import Any

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from src import config, db

logger = logging.getLogger(__name__)

_ANALYSE_MODULE: ModuleType | None = None
_ANALYSE_PATH = Path(__file__).resolve().parent.parent / "analyse-activities.py"


def _get_analyse_module() -> ModuleType:
    """Dynamically load analyse-activities.py (hyphenated filename)."""
    global _ANALYSE_MODULE
    if _ANALYSE_MODULE is None:
        spec = importlib.util.spec_from_file_location("analyse_activities", _ANALYSE_PATH)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load module from {_ANALYSE_PATH}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _ANALYSE_MODULE = module
    return _ANALYSE_MODULE


def run_sync(limit: int = 30, after_days: int | None = None) -> dict[str, Any]:
    """Execute Strava API sync and return summary counts."""
    try:
        module = _get_analyse_module()
        result = module.sync_recent_from_api(limit=limit, after_days=after_days)
        logger.info("Strava sync completed: %s", result)
        return result or {"fetched": 0, "db_inserted": 0}
    except Exception as exc:
        logger.exception("Strava sync failed: %s", exc)
        return {"error": str(exc), "fetched": 0, "db_inserted": 0}


def start_scheduler(run_on_startup: bool = True) -> BackgroundScheduler:
    """Seed DB from CSV, start APScheduler daily cron, and optionally run startup sync."""
    seeded = db.seed_from_csv()
    logger.info("Startup CSV seed complete (inserted %d rows).", seeded)

    scheduler = BackgroundScheduler()
    scheduler.add_job(
        run_sync,
        trigger=CronTrigger(hour=config.SYNC_CRON_HOUR, minute=0),
        id="daily_strava_sync",
        replace_existing=True,
    )

    if run_on_startup:
        scheduler.add_job(
            run_sync,
            trigger="date",
            run_date=datetime.now(),
            id="startup_strava_sync",
            replace_existing=True,
        )

    scheduler.start()
    logger.info(
        "Scheduler started (daily Strava sync at %02d:00, startup_sync=%s).",
        config.SYNC_CRON_HOUR,
        run_on_startup,
    )
    return scheduler


def stop_scheduler(scheduler: BackgroundScheduler | None) -> None:
    """Shut down the background scheduler cleanly."""
    if scheduler is not None and scheduler.running:
        scheduler.shutdown(wait=False)
        logger.info("Scheduler stopped.")
