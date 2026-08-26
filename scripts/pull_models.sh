#!/bin/sh
set -e
HOST="${OLLAMA_HOST:-http://127.0.0.1:11434}"
export OLLAMA_HOST="$HOST"
echo "Pulling Ollama models against $OLLAMA_HOST"
ollama pull llama3.2:3b
ollama pull nomic-embed-text
echo "Done."
