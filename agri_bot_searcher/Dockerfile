# Agriculture Bot Searcher - Docker Container
FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV OLLAMA_HOST=host.docker.internal:11434

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        wget \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY tests/ ./tests/
COPY scripts/ ./scripts/
COPY docs/ ./docs/

# Create a non-root user
RUN useradd --create-home --shell /bin/bash agribot
RUN chown -R agribot:agribot /app
USER agribot

# Expose port for potential web interface
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["python", "src/agriculture_chatbot.py", "--help"]

# Labels
LABEL maintainer="Agriculture Bot Team"
LABEL description="Multi-agent agriculture chatbot with web search and citations"
LABEL version="1.0.0"
