# Use Python 3.11 slim image
FROM python:3.11-slim

# Install system dependencies for OpenCV and Blender
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    # Blender dependencies
    wget \
    xz-utils \
    libxrender1 \
    libxi6 \
    libxkbcommon0 \
    libxxf86vm1 \
    libxfixes3 \
    libxext6 \
    libx11-6 \
    libsm6 \
    && rm -rf /var/lib/apt/lists/*

# Install Blender 4.2 LTS (stable, supports files from 2.8+)
# Note: Using 4.2 LTS for Docker stability; local dev uses system Blender
RUN wget -q https://download.blender.org/release/Blender4.2/blender-4.2.3-linux-x64.tar.xz \
    && tar -xf blender-4.2.3-linux-x64.tar.xz \
    && mv blender-4.2.3-linux-x64 /opt/blender \
    && rm blender-4.2.3-linux-x64.tar.xz

# Set Blender path environment variable
ENV BLENDER_PATH=/opt/blender/blender

# Set working directory
WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv sync --frozen --no-dev

# Copy application code
COPY . .

# Expose port (Cloud Run uses PORT environment variable)
ENV PORT=8080

# Set Python to unbuffered mode for immediate log output
ENV PYTHONUNBUFFERED=1

# Run the application
CMD [".venv/bin/uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--log-level", "info"]
