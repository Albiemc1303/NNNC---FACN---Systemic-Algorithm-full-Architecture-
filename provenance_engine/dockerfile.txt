# ---------- Base Image ----------
FROM python:3.11-slim AS base

# Prevent Python from writing .pyc files and enable unbuffered logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# ---------- Install Dependencies ----------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---------- Copy Application ----------
COPY provenance_engine/ ./provenance_engine/

# ---------- Environment Variables ----------
# Default token can be overridden via Fly secrets: fly secrets set PROVENANCE_API_TOKEN=yourtoken
ENV PROVENANCE_API_TOKEN="fly secrets set PROVENANCE_API_TOKEN=yourtoken"
ENV PORT=8080

# ---------- Expose Port ----------
EXPOSE 8080

# ---------- Run Application ----------
# Run via Uvicorn directly, with reload disabled (for production)
CMD ["uvicorn", "provenance_engine.app:app", "--host", "0.0.0.0", "--port", "8080"]
