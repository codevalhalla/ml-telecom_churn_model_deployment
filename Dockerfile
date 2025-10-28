# Use Python 3.8.12 slim as base
FROM python:3.12-slim

# Install pipenv
RUN pip install pipenv

# Set working directory
WORKDIR /app

# Copy dependency files first (for caching)
COPY ["Pipfile", "Pipfile.lock", "./"]

# Install dependencies directly into system Python
RUN pipenv install --system --deploy

# Copy application code
COPY ["predict.py", "model_C=0.1.bin", "./"]

# Expose port 9696
EXPOSE 9696

ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "predict:app"]   
