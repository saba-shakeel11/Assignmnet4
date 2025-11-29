# Optional, if custom image needed for components (e.g., if lightweight not sufficient)
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ ./src/
COPY pipeline.py .

CMD ["python", "pipeline.py"]