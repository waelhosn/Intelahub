FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

ARG INSTALL_RERANKER=0

COPY requirements.txt /app/requirements.txt
COPY requirements-rerank.txt /app/requirements-rerank.txt

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt && \
    if [ "$INSTALL_RERANKER" = "1" ]; then pip install --no-cache-dir -r /app/requirements-rerank.txt; fi

COPY app /app/app
COPY pyproject.toml README.md /app/

RUN mkdir -p /app/data/chroma

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
