# AGENTS.md — Strava AI Cycling Coach & MCP Server

## 1. Project Overview
This project ingests Strava cycling activities (from historical `.fit` exports and live Strava v3 API streams), computes physiological and power/heart-rate training metrics, stores the enriched dataset in a local SQLite database (`cycling.db`), and exposes the dataset alongside a personal training analysis (`playbook.md`) via a **FastMCP** server.

The goal is to give an AI coding/coaching client (Antigravity / Gemini) live access to:
1. Historical training correlations and personal responder rules (`playbook.md`).
2. Current fitness (CTL), fatigue (ATL), form (TSB), and rolling 90-day FTP.
3. Recent activity metrics (Normalized Power, Intensity Factor, Variability Index, zone distributions, HIIT surges, aerobic decoupling, climbing percentage).
4. On-demand and scheduled daily ingestion of new Strava activities.

---

## 2. Target Repository Structure
Maintain the following clean directory structure:

```text
.
├── AGENTS.md                   # Project specification and agent instructions
├── Dockerfile                  # Python 3.12-slim container definition
├── docker-compose.yml          # Container orchestration & ./server bind mount
├── requirements.txt            # Python package dependencies
├── .env.example                # Template environment variables
├── analyse-activities.py       # Core metric engine, FIT parser, and Strava API client
├── src/
│   ├── __init__.py
│   ├── config.py               # Central configuration & environment variables
│   ├── db.py                   # SQLite schema, CSV seeder, and CTL/ATL/TSB views
│   ├── scheduler.py            # APScheduler background worker for daily Strava sync
│   └── mcp_server.py           # FastMCP server exposing tools and resources over SSE/HTTP
└── server/                     # Bind-mounted into container at /app/server (git-ignored contents)
    ├── .gitkeep
    ├── playbook.md             # User's historical correlation analysis & training findings
    ├── enriched_activities.csv # Master enriched CSV dataset
    ├── .strava_tokens.json     # OAuth2 Client ID, Secret, Refresh & Access Tokens
    └── cycling.db              # Auto-generated SQLite database
