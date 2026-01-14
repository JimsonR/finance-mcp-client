FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (if needed for psutil)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port (Apply.build may use PORT env var)
EXPOSE 8001

# Health check - lightweight version
HEALTHCHECK --interval=60s --timeout=5s --start-period=10s --retries=2 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8001/health', timeout=3)" || exit 1

# Run application with resource-constrained settings
# - workers=1: Single worker to save memory
# - worker-connections=50: Limit concurrent connections
# - timeout-keep-alive=30: Faster connection cleanup
# - limit-concurrency=50: Prevent overload
# - no-access-log: Reduce I/O and memory
CMD ["uvicorn", "api_server:app", \
     "--host", "0.0.0.0", \
     "--port", "8001", \
     "--workers", "1", \
     "--limit-concurrency", "50", \
     "--timeout-keep-alive", "30", \
     "--no-access-log"]
