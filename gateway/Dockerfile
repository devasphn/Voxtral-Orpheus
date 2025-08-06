FROM python:3.10-slim
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code and the static UI file
COPY main.py .
COPY static /app/static

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
