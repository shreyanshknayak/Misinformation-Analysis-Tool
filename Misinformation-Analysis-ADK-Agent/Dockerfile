# 1. Use the same Python base you specified
FROM python:3.11-slim

# 2. Set the working directory
WORKDIR /app

# 3. Set environment variables for Python
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PORT 8080

# 4. Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# THE FIX: Install both gunicorn and uvicorn
RUN pip install gunicorn uvicorn

# 5. Copy your entire application code
COPY . .

# 6. Expose the port
EXPOSE 8080

# 7. THE FIX:
# -k uvicorn.workers.UvicornWorker tells Gunicorn to use the Uvicorn worker
CMD gunicorn --bind 0.0.0.0:8080 -k uvicorn.workers.UvicornWorker misinformation-app.agent:app