#!/usr/bin/env bash
#
# validate-submission.sh — Paper Formatter OpenEnv Submission Validator
#
# Checks:
#   1. HF Space is live and responds to /reset
#   2. Docker image builds successfully
#   3. openenv validate passes (or local yaml check)
#
# Usage:
#   ./validate-submission.sh <hf_space_url> [repo_dir]
#
# Example:
#   ./validate-submission.sh https://your-username-paper-formatter-openenv.hf.space .
#

set -uo pipefail

DOCKER_BUILD_TIMEOUT=600

if [ -t 1 ]; then
  RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
  BOLD='\033[1m'; NC='\033[0m'
else
  RED=''; GREEN=''; YELLOW=''; BOLD=''; NC=''
fi

run_with_timeout() {
  local secs="$1"; shift
  if command -v timeout &>/dev/null; then
    timeout "$secs" "$@"
  elif command -v gtimeout &>/dev/null; then
    gtimeout "$secs" "$@"
  else
    "$@" & local pid=$!
    ( sleep "$secs" && kill "$pid" 2>/dev/null ) & local watcher=$!
    wait "$pid" 2>/dev/null; local rc=$?
    kill "$watcher" 2>/dev/null; wait "$watcher" 2>/dev/null; return $rc
  fi
}

CLEANUP_FILES=()
cleanup() { rm -f "${CLEANUP_FILES[@]+"${CLEANUP_FILES[@]}"}"; }
trap cleanup EXIT

PING_URL="${1:-}"
REPO_DIR="${2:-.}"

if [ -z "$PING_URL" ]; then
  printf "Usage: %s <ping_url> [repo_dir]\n" "$0"
  printf "  ping_url   HuggingFace Space URL (e.g. https://user-paper-formatter-openenv.hf.space)\n"
  printf "  repo_dir   Path to repo (default: current directory)\n"
  exit 1
fi

if ! REPO_DIR="$(cd "$REPO_DIR" 2>/dev/null && pwd)"; then
  printf "Error: directory '%s' not found\n" "${2:-.}"; exit 1
fi

PING_URL="${PING_URL%/}"
PASS=0

log()  { printf "[%s] %b\n" "$(date -u +%H:%M:%S)" "$*"; }
pass() { log "${GREEN}PASSED${NC} -- $1"; PASS=$((PASS + 1)); }
fail() { log "${RED}FAILED${NC} -- $1"; }
hint() { printf "  ${YELLOW}Hint:${NC} %b\n" "$1"; }
stop_at() {
  printf "\n${RED}${BOLD}Validation stopped at %s.${NC} Fix the above before continuing.\n" "$1"
  exit 1
}

printf "\n${BOLD}============================================${NC}\n"
printf "${BOLD}  Paper Formatter OpenEnv — Validator${NC}\n"
printf "${BOLD}============================================${NC}\n"
log "Repo:     $REPO_DIR"
log "Ping URL: $PING_URL"
printf "\n"

# ── Step 1: HF Space health ──
log "${BOLD}Step 1/4: Pinging HF Space${NC} ($PING_URL/health) ..."
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$PING_URL/health" --max-time 30 2>/dev/null || echo "000")

if [ "$HTTP_CODE" = "200" ]; then
  pass "HF Space /health responds 200"
else
  fail "HF Space /health returned HTTP $HTTP_CODE (expected 200)"
  hint "Check your Space is running: open $PING_URL in browser"
  stop_at "Step 1"
fi

# ── Step 1b: /reset endpoint ──
log "${BOLD}Step 1b: Testing /reset endpoint${NC} ..."
RESET_CODE=$(curl -s -o /dev/null -w "%{http_code}" -X POST \
  -H "Content-Type: application/json" \
  -d '{"task_id":"task_easy"}' \
  "$PING_URL/reset" --max-time 30 2>/dev/null || echo "000")

if [ "$RESET_CODE" = "200" ]; then
  pass "/reset returns 200"
else
  fail "/reset returned HTTP $RESET_CODE"
  hint "Ensure task_easy is implemented in tasks.py"
  stop_at "Step 1b"
fi

# ── Step 2: openenv.yaml check ──
log "${BOLD}Step 2/4: Checking openenv.yaml${NC} ..."
YAML_FILE="$REPO_DIR/openenv.yaml"
if [ ! -f "$YAML_FILE" ]; then
  fail "openenv.yaml not found in $REPO_DIR"
  stop_at "Step 2"
fi

# Check required keys
REQUIRED_KEYS="name version description tasks endpoints"
for key in $REQUIRED_KEYS; do
  if grep -q "^${key}:" "$YAML_FILE"; then
    : # ok
  else
    fail "openenv.yaml missing required key: $key"
    stop_at "Step 2"
  fi
done

# Check 3+ tasks
TASK_COUNT=$(grep -c "^  - id:" "$YAML_FILE" || echo 0)
if [ "$TASK_COUNT" -ge 3 ]; then
  pass "openenv.yaml valid with $TASK_COUNT tasks"
else
  fail "openenv.yaml has only $TASK_COUNT tasks (need >= 3)"
  stop_at "Step 2"
fi

# ── Step 3: Local tests ──
log "${BOLD}Step 3/4: Running local test suite${NC} ..."
if command -v python3 &>/dev/null && [ -f "$REPO_DIR/tests/test_environment.py" ]; then
  TEST_OUTPUT=$(cd "$REPO_DIR" && python3 tests/test_environment.py 2>&1)
  if echo "$TEST_OUTPUT" | grep -q "ALL TESTS PASSED"; then
    PASSED=$(echo "$TEST_OUTPUT" | grep "Results:" | grep -o "[0-9]*/[0-9]*")
    pass "All local tests passed ($PASSED)"
  else
    fail "Some tests failed"
    echo "$TEST_OUTPUT" | tail -20
    stop_at "Step 3"
  fi
else
  log "  ${YELLOW}Skipping local tests${NC} (python3 not found or test file missing)"
fi

# ── Step 4: Docker build ──
log "${BOLD}Step 4/4: Running docker build${NC} ..."
if ! command -v docker &>/dev/null; then
  fail "docker not found — install from https://docs.docker.com/get-docker/"
  stop_at "Step 4"
fi

DOCKERFILE=""
if [ -f "$REPO_DIR/Dockerfile" ]; then
  DOCKERFILE="$REPO_DIR"
elif [ -f "$REPO_DIR/server/Dockerfile" ]; then
  DOCKERFILE="$REPO_DIR/server"
else
  fail "No Dockerfile found in $REPO_DIR or $REPO_DIR/server"
  stop_at "Step 4"
fi

BUILD_OK=false
BUILD_OUTPUT=$(run_with_timeout "$DOCKER_BUILD_TIMEOUT" \
  docker build -t paper-formatter-env-test "$DOCKERFILE" 2>&1) && BUILD_OK=true

if [ "$BUILD_OK" = true ]; then
  pass "Docker build succeeded"
  # Clean up image
  docker rmi paper-formatter-env-test >/dev/null 2>&1 || true
else
  fail "Docker build failed (timeout=${DOCKER_BUILD_TIMEOUT}s)"
  echo "$BUILD_OUTPUT" | tail -25
  stop_at "Step 4"
fi

printf "\n${BOLD}============================================${NC}\n"
printf "${GREEN}${BOLD}  All 4/4 checks passed!${NC}\n"
printf "${GREEN}${BOLD}  Your submission is ready to submit.${NC}\n"
printf "${BOLD}============================================${NC}\n\n"
exit 0
