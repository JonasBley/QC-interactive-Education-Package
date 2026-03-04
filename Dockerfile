# Use a lightweight, stable Python base image
FROM python:3.13-slim

# Install system-level graphics dependencies required by Matplotlib and Pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    fonts-liberation \
    && rm -rf /var/lib/apt/lists/*

# Force Matplotlib to use the headless 'Agg' backend to prevent UI thread crashes
ENV MPLBACKEND=Agg

# Establish the working directory inside the container
WORKDIR /app

# Transfer and install dependencies first to leverage Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application (notebooks, source code)
COPY . .

# Expose the standard HTTP port
EXPOSE 8080

# Execute Voilà bound to all network interfaces (0.0.0.0) so the cloud router can access it
CMD ["voila", "--no-browser", "--port=8080", "--Voila.ip=0.0.0.0", "--theme=light", "app.ipynb"]