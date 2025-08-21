# Use Python base image
FROM python:3.10

# Prevent Python from buffering and writing pyc
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system-level dependencies (updated package names for Debian Trixie)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy everything into container
COPY . .

# Expose Streamlit's default port
EXPOSE 7860

# Health check (optional)
HEALTHCHECK CMD curl --fail http://localhost:7860/_stcore/health || exit 1

# Run app
CMD ["streamlit", "run", "src/app.py", "--server.port=7860", "--server.address=0.0.0.0"]
