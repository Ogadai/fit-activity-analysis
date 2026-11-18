# Strava AI Cycling Coach & MCP Server

Ingests Strava cycling activities (from historical `.fit` / `.fit.gz` bulk exports and live Strava v3 API streams), computes power/heart-rate training metrics, stores them in SQLite (`server/cycling.db`) and CSV (`server/enriched_activities.csv`), and exposes your dataset alongside `server/playbook.md` over a **FastMCP** SSE server.

---

## Data Directory Layout (`./server/`)

When running locally, `DATA_DIR` defaults to `./server` (in Docker it is mounted at `/app/server`):

- `server/strava_export/` — Unzipped Strava bulk archive (`activities.csv` + FIT files) for initial historical ingestion
- `server/enriched_activities.csv` — Master enriched activity dataset
- `server/cycling.db` — Auto-generated SQLite database (seeded from `enriched_activities.csv` and updated by API syncs)
- `server/playbook.md` — Your personal training analysis and responder rules
- `server/.strava_tokens.json` — Strava OAuth2 credentials and cached tokens (`client_id`, `client_secret`, `refresh_token`)

---

## Initial Ingestion of Exported Strava History

1. Request and download your **Strava account archive** (Bulk Export), then extract it so that `activities.csv` and the activity files sit under `server/strava_export/`:
   ```text
   server/strava_export/
   ├── activities.csv
   └── activities/
       ├── 1234567890.fit.gz
       └── ...
   ```
   *(If your export is in a different folder, set the `STRAVA_DIR` environment variable to that path.)*

2. Run the bulk parser to enrich all historical activities:
   ```powershell
   python analyse-activities.py --mode bulk
   ```
   This sorts `activities.csv` chronologically, parses `.fit` / `.fit.gz` streams, computes rolling 90-day `effective_ftp`, 365-day `effective_max_hr`, Normalized Power, IF, VI, zones, peak powers, HIIT surges, aerobic decoupling, and climbing percentage, and writes `server/enriched_activities.csv`.

3. Seed the SQLite database from the generated CSV (or start the MCP server, which seeds automatically on startup):
   ```powershell
   python -c "from src import db; print('Inserted rows:', db.seed_from_csv())"
   ```

---

## Running Locally on Windows (Without Docker)

### 1. Create a Virtual Environment & Install Dependencies
From the repository root in PowerShell:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Configure Environment Variables (Optional)
`src/config.py` reads standard OS environment variables (it does not auto-load `.env` files). Defaults work out of the box (`DATA_DIR=./server`, `MCP_PORT=8050`, `FTP=250`, `MAX_HR=185`, `SYNC_CRON_HOUR=2`).

To override any setting in PowerShell for the current session:
```powershell
$env:FTP = "250"
$env:MAX_HR = "185"
$env:MCP_PORT = "8050"
```

### 3. Ingest from History Export (Bulk)
If you haven't generated `server/enriched_activities.csv` yet:
```powershell
python analyse-activities.py --mode bulk
```

### 4. Ingest from the Strava API
Provide Strava OAuth2 credentials either in `server/.strava_tokens.json`:
```json
{
  "client_id": "YOUR_CLIENT_ID",
  "client_secret": "YOUR_CLIENT_SECRET",
  "refresh_token": "YOUR_REFRESH_TOKEN"
}
```
or via PowerShell environment variables:
```powershell
$env:STRAVA_CLIENT_ID = "YOUR_CLIENT_ID"
$env:STRAVA_CLIENT_SECRET = "YOUR_CLIENT_SECRET"
$env:STRAVA_REFRESH_TOKEN = "YOUR_REFRESH_TOKEN"
```

Then fetch and enrich recent activities:
```powershell
python analyse-activities.py --mode api --limit 30
```
Optional flags:
- `--limit N` — Max recent activities to inspect (default: `30`)
- `--after-days N` — Only check activities from the last `N` days

API mode appends new activities to `server/enriched_activities.csv` and inserts them directly into `server/cycling.db`.

### 5. Updating the SQLite Database (`server/cycling.db`)
- **Incremental CSV seed:** If you ran `--mode bulk` or updated `server/enriched_activities.csv`, import any missing `Activity ID` rows into `server/cycling.db`:
  ```powershell
  python -c "from src import db; print('Inserted rows:', db.seed_from_csv())"
  ```
- **Full rebuild:** To rebuild `server/cycling.db` from scratch from `server/enriched_activities.csv`, delete `server/cycling.db` and re-run the command above.

### 6. Start the MCP Server Locally
Optionally place your `playbook.md` in `server/playbook.md`, then start the server:
```powershell
python -m src.mcp_server
```
On startup, the server:
1. Seeds `server/cycling.db` from `server/enriched_activities.csv`.
2. Starts the APScheduler background worker (runs an immediate Strava API sync and schedules daily syncs at `SYNC_CRON_HOUR`, default `02:00`).
3. Serves the FastMCP SSE endpoint at `http://localhost:8050/sse`.

### 7. Connect Your MCP Client (Antigravity)
Add the local SSE URL to your Antigravity MCP settings:
```text
http://localhost:8050/sse
```

Available MCP tools and resources:
- **Tools:** `get_training_playbook`, `get_current_readiness`, `get_recent_activities`, `sync_strava_now`, `query_training_database`
- **Resources:** `coach://playbook`, `coach://status-summary`

---

## Docker Deployment (Debian Server)

1. Copy your files into `./server/` on the Debian host:
   - `server/enriched_activities.csv` *(or place your unzipped Strava export in `server/strava_export/` to generate it inside Docker — see step 3a)*
   - `server/playbook.md`
   - `server/.strava_tokens.json`
2. Optionally create a `.env` file from `.env.example` to configure `FTP`, `MAX_HR`, `MCP_PORT`, `SYNC_CRON_HOUR`, or Strava credentials.
3. Build and start the container:
   ```bash
   docker compose up -d --build
   ```
   **3a. Optional — Run initial bulk history ingestion inside Docker:**  
   If you placed your unzipped Strava bulk export at `./server/strava_export/` instead of copying a pre-built `enriched_activities.csv`, run the bulk ingestion in a one-off container before starting the service (or `exec` into the running container):
   ```bash
   # Before `docker compose up -d`:
   docker compose run --rm strava-coach python analyse-activities.py --mode bulk
   docker compose up -d --build
   ```
   If the container is already running, run bulk ingestion and then seed/restart so `cycling.db` picks up the new CSV:
   ```bash
   docker compose exec strava-coach python analyse-activities.py --mode bulk
   docker compose exec strava-coach python -c "from src import db; print('Inserted rows:', db.seed_from_csv())"
   ```
4. Add the MCP server URL (`http://<debian-server-ip>:8050/sse`) to your Antigravity MCP settings.
