FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml README.md LICENSE main.py ./
COPY app ./app
RUN pip install --no-cache-dir .

EXPOSE 8000

CMD ["phone-chat", "--host", "0.0.0.0", "--port", "8000"]

