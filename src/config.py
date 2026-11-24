"""Central configuration and environment variable loading."""

import os
from pathlib import Path

# Cold-start physiology fallback settings (used only when no historical data exists)
DEFAULT_FTP: int = int(os.getenv("FTP", "250"))
DEFAULT_MAX_HR: int = int(os.getenv("MAX_HR", "185"))
FTP: int = DEFAULT_FTP
MAX_HR: int = DEFAULT_MAX_HR

# Automated eFTP (Morton 3-parameter Critical Power model, Intervals.icu-style)
EFTP_MIN_DURATION_SEC: int = int(os.getenv("EFTP_MIN_DURATION_SEC", "180"))
EFTP_WINDOW_DAYS: int = int(os.getenv("EFTP_WINDOW_DAYS", "90"))

# Storage & data paths
DATA_DIR: Path = Path(os.getenv("DATA_DIR", "./server"))
STRAVA_DIR: Path = Path(os.getenv("STRAVA_DIR", str(DATA_DIR / "strava_export")))
DB_PATH: Path = Path(os.getenv("DB_PATH", str(DATA_DIR / "cycling.db")))
CSV_PATH: Path = Path(os.getenv("CSV_PATH", str(DATA_DIR / "enriched_activities.csv")))
PLAYBOOK_PATH: Path = Path(os.getenv("PLAYBOOK_PATH", str(DATA_DIR / "playbook.md")))
TOKEN_FILE: Path = Path(os.getenv("TOKEN_FILE", str(DATA_DIR / ".strava_tokens.json")))

# MCP server & scheduler settings
MCP_HOST: str = os.getenv("MCP_HOST", "0.0.0.0")
MCP_PORT: int = int(os.getenv("MCP_PORT", "8050"))
SYNC_CRON_HOUR: int = int(os.getenv("SYNC_CRON_HOUR", "2"))
SYNC_SOURCE: str = os.getenv("SYNC_SOURCE", "auto").strip().lower()

# Intervals.icu API credentials (Athlete ID '0' resolves to the API key owner)
INTERVALS_API_KEY: str = os.getenv("INTERVALS_API_KEY", "")
INTERVALS_ATHLETE_ID: str = os.getenv("INTERVALS_ATHLETE_ID", "0")

# Strava API credentials (optional if provided in .strava_tokens.json)
STRAVA_CLIENT_ID: str = os.getenv("STRAVA_CLIENT_ID", "")
STRAVA_CLIENT_SECRET: str = os.getenv("STRAVA_CLIENT_SECRET", "")
STRAVA_REFRESH_TOKEN: str = os.getenv("STRAVA_REFRESH_TOKEN", "")
