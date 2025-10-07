FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
COPY services/ai-ml/mlops_requirements.txt ./mlops_requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r mlops_requirements.txt

# Copy application code
COPY . .

# Create directories for MLOps
RUN mkdir -p /app/models/registry /app/models/experiments /app/models/monitoring /app/models/mlruns

# Set environment variables for MLOps
ENV PYTHONPATH=/app
ENV MLFLOW_TRACKING_URI=file:///app/models/mlruns
ENV MLOPS_BASE_PATH=/app/models

# Expose ports
EXPOSE 8005
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8005/health || exit 1

# Run the application
CMD ["python", "services/ai-ml/main.py"]