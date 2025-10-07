# Strategy Engine Service Dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy shared modules
COPY ../../shared /app/shared

# Copy service code
COPY . /app

# Create non-root user
RUN useradd -m -u 1000 strategy-engine && chown -R strategy-engine:strategy-engine /app
USER strategy-engine

# Expose port
EXPOSE 8007

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8007/health || exit 1

# Run the service
CMD ["python", "main.py"]