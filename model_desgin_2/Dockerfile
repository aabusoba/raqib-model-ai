# Use Python 3.9 as base
FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose ports
EXPOSE 8502
EXPOSE 5001

# Healthcheck for Streamlit (Design 2 uses 8502)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl --fail http://localhost:8502/_stcore/health || exit 1

# Default entrypoint
CMD ["streamlit", "run", "app.py", "--server.port=8502", "--server.address=0.0.0.0"]
