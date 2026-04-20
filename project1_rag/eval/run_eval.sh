#!/bin/bash
# Run the promptfoo eval with the correct Python venv and env vars.
#
# Usage:
#   ./run_eval.sh                     # run all 30 questions × 5 configs
#   ./run_eval.sh --filter-first-n 3  # run first 3 questions only (quick test)
#
# After running, open the dashboard:
#   promptfoo view

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ENV_FILE="$(dirname "$PROJECT_ROOT")/.env"

# Export API keys from .env so promptfoo's LLM judge can use them.
export ANTHROPIC_API_KEY=$(grep ANTHROPIC_API_KEY "$ENV_FILE" | cut -d= -f2)
export OPENAI_API_KEY=$(grep OPENAI_API_KEY "$ENV_FILE" | cut -d= -f2)
export SUPABASE_DB_URL=$(grep SUPABASE_DB_URL "$ENV_FILE" | cut -d= -f2-)

# Tell promptfoo to use our venv's Python (has all dependencies).
export PROMPTFOO_PYTHON="$PROJECT_ROOT/../.venv/bin/python"

cd "$SCRIPT_DIR"

# -j 1: one request at a time (avoids Anthropic 5 req/min rate limit)
# --delay 15000: 15s between requests (4 provider calls + 1 grading call per question)
promptfoo eval --no-cache -j 1 --delay 15000 "$@"
