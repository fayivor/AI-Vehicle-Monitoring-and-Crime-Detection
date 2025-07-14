# Multi-stage Docker build for AI Vehicle Monitoring System
# Stage 1: Dependencies and model downloads
FROM python:3.11-slim as dependencies

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements/base.txt requirements/prod.txt ./requirements/
RUN pip install --no-cache-dir -r requirements/prod.txt

# Download and cache models
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# Stage 2: Application code and configs
FROM dependencies as application

# Copy application code
COPY src/ ./src/
COPY pyproject.toml ./

# Install the application
RUN pip install -e .

# Stage 3: Production runtime with minimal footprint
FROM python:3.11-slim as production

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

# Install only runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Create app directory and set ownership
WORKDIR /app
RUN chown -R appuser:appuser /app

# Copy from previous stages
COPY --from=application --chown=appuser:appuser /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=application --chown=appuser:appuser /usr/local/bin /usr/local/bin
COPY --from=application --chown=appuser:appuser /app/src ./src
COPY --from=application --chown=appuser:appuser /app/pyproject.toml ./

# Create data directories
RUN mkdir -p data/faiss_index data/chromadb logs && \
    chown -R appuser:appuser data logs

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["gunicorn", "src.api.main:app", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "-b", "0.0.0.0:8000"]
