**Deploy on the Debian server and wire up VS Code Antigravity.
1. Copy your three files into `./server/` on the Debian host:
   * `server/enriched_activities.csv`[cite: 1]
   * `server/playbook.md`
   * `server/.strava_tokens.json`
2. Run `docker compose up -d --build` on the Debian server.
3. Add the MCP server URL (`http://<debian-server-ip>:8050/sse`) to your Antigravity MCP settings on your PC.