# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set environment variables
# PYTHONDONTWRITEBYTECODE: Prevents Python from writing pyc files to disc
# PYTHONUNBUFFERED: Prevents Python from buffering stdout and stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory in the container
WORKDIR /app

# Install system dependencies (needed for some python packages like reportlab or scikit-learn)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container at /app
COPY requirements.txt /app/

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . /app/

# Expose port (default Flask port is 5000, but we can configure it)
EXPOSE 5000

# Run the application using Gunicorn
# app.app:app references the 'app' object in 'app/app.py'
# But wait, the app is in 'app/app.py', so the module path is app.app
# We need to make sure the python path is correct.
# Given the structure:
# /app
#   /app (directory)
#      app.py
# setting PYTHONPATH is useful or running from root.

ENV PYTHONPATH=/app

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app.app:app"]
