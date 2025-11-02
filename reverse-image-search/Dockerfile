# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code (main.py and your tool package)
COPY . .

# Environment variable set by Cloud Run specifying the port to listen on
ENV PORT 8080

# Command to run the application using uvicorn
# It listens on all interfaces (0.0.0.0) and on the port specified by $PORT
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]