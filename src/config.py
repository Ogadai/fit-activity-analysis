"""Central configuration and environment variable loading."""

import os
from pathlib import Path

# Physiology settings
FTP: int = int(os.getenv("FTP", "250"))
MAX_HR: int = int(os.getenv("MAX_HR", "185"))

# Storage & data paths
DATA_DIR: Path = Path(os.getenv("DATA_DIR", "./server"))
DB_PATH: Path = Path(os.getenv("DB_PATH", str(DATA_DIR / "cycling.db")))
CSV_PATH: Path = Path(os.getenv("CSV_PATH", str(DATA_DIR / "enriched_activities.csv")))
PLAYBOOK_PATH: Path = Path(os.getenv("PLAYBOOK_PATH", str(DATA_DIR / "playbook.md")))
TOKEN_FILE: Path = Path(os.getenv("TOKEN_FILE", str(DATA_DIR / ".strava_tokens.json")))

# MCP server & scheduler settings
MCP_HOST: str = os.getenv("MCP_HOST", "0.0.0.0")
MCP_PORT: int = int(os.getenv("MCP_PORT", "8050"))
SYNC_CRON_HOUR: int = int(os.getenv("SYNC_CRON_HOUR", "2"))

# Strava API credentials (optional if provided in .strava_tokens.json)
STRAVA_CLIENT_ID: str = os.getenv("STRAVA_CLIENT_ID", "")
STRAVA_CLIENT_SECRET: str = os.getenv("STRAVA_CLIENT_SECRET", "")
STRAVA_REFRESH_TOKEN: str = os.getenv("STRAVA_REFRESH_TOKEN", "")
