FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DATA_DIR=/app/server \
    MCP_PORT=8050

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY analyse-activities.py .
COPY src/ ./src/

EXPOSE 8050

CMD ["python", "-m", "src.mcp_server"]
