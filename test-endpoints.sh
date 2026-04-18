#!/usr/bin/env bash
# ── Lumos endpoint smoke test ─────────────────────────────────────
# Usage:
#   ./test-endpoints.sh [base_url]
#
#   Base URL defaults to http://localhost:10000.
#   Deployed URL: https://lumos.aws.lumiralabs.com
#
# Required env var:
#   LUMOS_API_KEY — the token the server checks on auth'd routes
#
# Example:
#   LUMOS_API_KEY=xxx ./test-endpoints.sh
#   LUMOS_API_KEY=xxx ./test-endpoints.sh https://lumos.aws.lumiralabs.com

set -u
BASE="${1:-http://localhost:10000}"
: "${LUMOS_API_KEY:?Set LUMOS_API_KEY first}"

pass=0
fail=0

hit() {
  local name="$1" method="$2" path="$3" expected="$4"; shift 4
  printf "  %-30s " "$name"
  status=$(curl -sS -o /tmp/lumos-resp.txt -w "%{http_code}" -X "$method" "$BASE$path" "$@" || echo "ERR")
  if [[ "$status" == "$expected" ]]; then
    echo "✓ $status"; pass=$((pass+1))
  else
    echo "✗ expected $expected got $status"
    echo "     body: $(head -c 200 /tmp/lumos-resp.txt)"
    fail=$((fail+1))
  fi
}

echo "Base URL: $BASE"
echo

echo "── Public endpoints (no auth required) ──"
hit "GET /"            GET  "/"        200
hit "GET /healthz"     GET  "/healthz" 200

echo
echo "── Auth checks (valid body, missing key must 401) ──"
# Note: FastAPI validates the Pydantic body before the decorator runs,
# so we must send a schema-valid payload to actually hit the auth check.
hit "POST /generate (no key)" POST "/generate" 401 \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"hi"}],"response_schema":null}'
hit "POST /embed (no key)"    POST "/embed"    401 \
  -H "Content-Type: application/json" \
  -d '{"inputs":"hi"}'

echo
echo "── Authed endpoints (with key) ──"
hit "POST /embed (valid key)" POST "/embed" 200 \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $LUMOS_API_KEY" \
  -d '{"inputs":"hello world","model":"text-embedding-3-small"}'

hit "POST /generate (valid key)" POST "/generate" 200 \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $LUMOS_API_KEY" \
  -d '{
    "messages":[{"role":"user","content":"say hi"}],
    "response_schema":{"type":"object","properties":{"greeting":{"type":"string"}},"required":["greeting"]},
    "model":"gpt-4o-mini"
  }'

# ── Book parser (only if PDF_PATH is set) ────────────────────────
if [[ -n "${PDF_PATH:-}" && -f "$PDF_PATH" ]]; then
  echo
  echo "── Book parser (using $PDF_PATH) ──"
  hit "POST /book/parse-pdf (file upload)" POST "/book/parse-pdf" 200 \
    -H "X-API-Key: $LUMOS_API_KEY" \
    -F "file=@$PDF_PATH"
fi

echo
echo "── Summary: $pass passed, $fail failed ──"
exit $fail
