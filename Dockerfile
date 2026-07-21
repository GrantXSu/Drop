FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir .

RUN useradd --create-home tracker && mkdir -p /app/data && chown tracker:tracker /app/data
USER tracker

CMD ["drop-tracker"]
