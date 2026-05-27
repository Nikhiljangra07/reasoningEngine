#!/usr/bin/env bash
# refresh_graph.sh — rebuild or update the project's code knowledge graph.
#
# Graphify lives at vendor/graphify/ (assimilated into this repo, MIT-licensed,
# v0.8.18 from github.com/safishamsi/graphify v8). Installed editable via:
#     pip install -e vendor/graphify --break-system-packages
#
# Output: graphify-out/graph.json — the file GraphifyAdapter reads.
#
# Strategy:
#   - graph.json missing → first-time `graphify extract`. Runs AST extraction
#     (free) AND a semantic pass via whichever LLM API key is set in .env
#     (Gemini is preferred for cost — set GEMINI_API_KEY / GOOGLE_API_KEY).
#     --no-cluster skips the post-extraction clustering step.
#   - graph.json present → `graphify update`. AST-only, no LLM call, free.
#
# Pass --full or --cluster to enable clustering.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Source .env so graphify finds the API keys. set -a auto-exports every
# var defined while it's active. Safe for our KEY=value lines.
if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  . ./.env
  set +a
fi

if ! command -v graphify >/dev/null 2>&1; then
  echo "graphify CLI not found on PATH." >&2
  echo "Install the vendored copy:" >&2
  echo "  pip install -e vendor/graphify --break-system-packages" >&2
  exit 1
fi

echo "→ graphify $(graphify --version 2>/dev/null || echo '(no --version)')"

# Tell the user which API key graphify will pick up. Don't echo the value.
for key in GEMINI_API_KEY GOOGLE_API_KEY ANTHROPIC_API_KEY OPENAI_API_KEY MOONSHOT_API_KEY DEEPSEEK_API_KEY; do
  if [ -n "${!key:-}" ]; then
    echo "→ $key present"
  fi
done

if [ -f graphify-out/graph.json ]; then
  echo "→ graph.json exists — running 'graphify update' (AST-only, no LLM call)"
  exec graphify update . --no-cluster
fi

echo "→ no existing graph; running first-time 'graphify extract' with --no-cluster"
if [ -n "${GRAPHIFY_GEMINI_MODEL:-}" ]; then
  echo "  (semantic step runs via gemini/${GRAPHIFY_GEMINI_MODEL})"
else
  echo "  (semantic step runs via the default backend — set GRAPHIFY_GEMINI_MODEL to the cheapest model)"
fi
FLAGS=("--no-cluster")
for arg in "$@"; do
  case "$arg" in
    --full|--cluster) FLAGS=() ;;
    *) FLAGS+=("$arg") ;;
  esac
done
exec graphify extract . "${FLAGS[@]}"
