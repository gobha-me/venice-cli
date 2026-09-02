FROM python:3.14-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1
WORKDIR /src
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN python -m venv /opt/venice \
    && /opt/venice/bin/python -m pip install --upgrade pip \
    && /opt/venice/bin/python -m pip install ".[all]"

FROM python:3.14-slim

LABEL org.opencontainers.image.source="https://github.com/gobha-me/venice-cli" \
      org.opencontainers.image.licenses="MIT"

RUN groupadd --system --gid 10001 venice \
    && useradd --uid 10001 --gid venice --no-create-home \
       --home-dir /nonexistent --shell /usr/sbin/nologin venice \
    && install -d -o venice -g venice /srv/venice
COPY --from=builder /opt/venice /opt/venice

ENV HOME=/tmp/venice-home \
    PATH=/opt/venice/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
WORKDIR /srv/venice
USER 10001:10001
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=2).read()"]

ENTRYPOINT ["venice", "mcp-serve", "--http", "--host", "0.0.0.0", "--port", "8000"]
