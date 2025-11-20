# Base image: Python 3.11 slim to save space
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Upgrade pip and install all dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose ports
# FastAPI default: 8000
# Streamlit default: 8501
EXPOSE 8001
EXPOSE 8501

# Run both services
# Use "sh -c" to run both commands in background
CMD sh -c "uvicorn app.main:app --host 0.0.0.0 --port 8001 & streamlit run app/streamlit_app.py --server.port 8501 --server.address 0.0.0.0"
