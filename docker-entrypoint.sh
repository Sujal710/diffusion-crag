#!/bin/bash
# Starts the two processes this app needs inside one container:
#   1. the MCP server (web search tool) on 127.0.0.1:8001, only reachable from inside the container
#   2. the API + web page on 0.0.0.0:8000
# If either one stops, the container exits so Docker's restart policy can bring it back.

python -m crag_app.mcp_server &
uvicorn crag_app.api:app --host 0.0.0.0 --port 8000 &

# docker stop sends SIGTERM to this script: pass it on to both processes.
trap 'kill $(jobs -p) 2>/dev/null; exit 0' TERM INT

wait -n          # returns as soon as the first of the two exits
code=$?
echo "A process exited (code $code); stopping the container." >&2
kill $(jobs -p) 2>/dev/null
exit "$code"
