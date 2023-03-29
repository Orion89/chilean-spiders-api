FROM python:3.9-slim

WORKDIR /app

COPY ./requirements.txt /app/requirements.txt

RUN pip install --upgrade pip
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

COPY main.py /app
COPY data /app/data
COPY feature_extractors /app/feature_extractors
COPY file_paths /app/file_paths
COPY schemas /app/schemas
COPY static /app/static
COPY utils /app/utils

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

