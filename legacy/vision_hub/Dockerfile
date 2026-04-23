FROM python:3.10-slim

# Install system dependencies for RealSense and OpenCV
RUN apt-get update && apt-get install -y \
    libusb-1.0-0 \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port for the dashboard/API
EXPOSE 8000

# Run the application
CMD ["python", "-m", "app.main"]
