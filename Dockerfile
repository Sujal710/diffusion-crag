# CRAG Assistant: FastAPI server + web page, Pinecone / Groq / Tavily / Cloudflare via API keys.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    FASTEMBED_CACHE_PATH=/opt/fastembed_cache

# OpenCV (pulled in by the OCR engine) needs these shared libraries.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Run as a normal user, not root.
RUN useradd --create-home --uid 1000 app
WORKDIR /app

# Dependencies first, so code changes don't reinstall them.
COPY requirements.txt .
RUN pip install -r requirements.txt

# Download the embedding model now, so the first upload or question doesn't wait for it.
RUN mkdir -p "$FASTEMBED_CACHE_PATH" \
    && python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5')"

COPY crag_app ./crag_app
COPY docker-entrypoint.sh ./docker-entrypoint.sh

# Folders the app writes to: user accounts (data) and generated videos (outputs).
RUN chmod +x docker-entrypoint.sh \
    && mkdir -p crag_app/data crag_app/outputs \
    && chown -R app:app /app/crag_app/data /app/crag_app/outputs "$FASTEMBED_CACHE_PATH"
VOLUME ["/app/crag_app/data", "/app/crag_app/outputs"]

USER app
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import socket, urllib.request; socket.create_connection(('127.0.0.1', 8001), 3).close(); urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"

# Starts the MCP server (web search) and the API together; see docker-entrypoint.sh.
# Keys come from the environment (docker run --env-file .env, or docker compose), never from the image.
CMD ["./docker-entrypoint.sh"]
