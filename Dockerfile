FROM python:3.11-slim

WORKDIR /app

# Install dependencies first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY web/ ./web/
COPY server.py run.py ./

# Non-root user for security
RUN useradd --create-home --shell /bin/bash lora && \
    chown -R lora:lora /app
USER lora

ENV PORT=8100 \
    HOST=0.0.0.0 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 8100

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; urllib.request.urlopen('http://localhost:'+__import__('os').environ.get('PORT','8100')+'/health').read(); sys.exit(0)" || exit 1

CMD ["python", "server.py"]
