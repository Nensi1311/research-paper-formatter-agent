#!/usr/bin/env bash
#
# validate-submission.sh — ScholarEnv Pre-Submission Validator
# Mirrors the OFFICIAL hackathon 3-step script exactly.
#
# Usage: ./validate-submission.sh <hf_space_url> [repo_dir]
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
  else
    "$@" & local pid=$!
    ( sleep "$secs" && kill "$pid" 2>/dev/null ) & local watcher=$!
    wait "$pid" 2>/dev/null; local rc=$?
    kill "$watcher" 2>/dev/null; wait "$watcher" 2>/dev/null; return $rc
  fi
}

PING_URL="${1:-}"
REPO_DIR="${2:-.}"
[ -z "$PING_URL" ] && { printf "Usage: %s <hf_space_url> [repo_dir]\n" "$0"; exit 1; }
REPO_DIR="$(cd "$REPO_DIR" 2>/dev/null && pwd)" || { printf "Error: dir not found\n"; exit 1; }
PING_URL="${PING_URL%/}"

log()  { printf "[%s] %b\n" "$(date -u +%H:%M:%S)" "$*"; }
pass() { log "${GREEN}PASSED${NC} — $1"; }
fail() { log "${RED}FAILED${NC} — $1"; }
hint() { printf "  ${YELLOW}Hint:${NC} %b\n" "$1"; }
stop_at() { printf "\n${RED}${BOLD}Stopped at %s.${NC}\n" "$1"; exit 1; }

printf "\n${BOLD}========================================${NC}\n"
printf "${BOLD}  ScholarEnv Pre-Submission Validator${NC}\n"
printf "${BOLD}========================================${NC}\n\n"

# Step 1: HF Space /reset returns 200
log "${BOLD}Step 1/3: HF Space /reset endpoint${NC} ..."
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" -X POST \
  -H "Content-Type: application/json" \
  -d '{"task_id":"formatting_compliance"}' \
  "$PING_URL/reset" --max-time 30 2>/dev/null || echo "000")

if [ "$HTTP_CODE" = "200" ]; then
  pass "/reset returns 200"
else
  fail "/reset returned HTTP $HTTP_CODE (expected 200)"
  hint "Make sure your Space is running and the URL is correct."
  hint "Try opening $PING_URL in your browser first."
  stop_at "Step 1"
fi

# Step 2: Docker build
log "${BOLD}Step 2/3: Docker build${NC} ..."
if ! command -v docker &>/dev/null; then
  fail "docker not found"
  hint "Install Docker: https://docs.docker.com/get-docker/"
  stop_at "Step 2"
fi
if [ -f "$REPO_DIR/Dockerfile" ]; then
  DOCKER_CONTEXT="$REPO_DIR"
elif [ -f "$REPO_DIR/server/Dockerfile" ]; then
  DOCKER_CONTEXT="$REPO_DIR/server"
else
  fail "No Dockerfile found"
  stop_at "Step 2"
fi

BUILD_OK=false
BUILD_OUTPUT=$(run_with_timeout "$DOCKER_BUILD_TIMEOUT" docker build "$DOCKER_CONTEXT" 2>&1) && BUILD_OK=true
if [ "$BUILD_OK" = true ]; then
  pass "Docker build succeeded"
else
  fail "Docker build failed"
  printf "%s\n" "$BUILD_OUTPUT" | tail -20
  stop_at "Step 2"
fi

# Step 3: openenv validate
log "${BOLD}Step 3/3: openenv validate${NC} ..."
if ! command -v openenv &>/dev/null; then
  fail "openenv not found — install: pip install openenv-core"
  stop_at "Step 3"
fi
VALIDATE_OK=false
VALIDATE_OUTPUT=$(cd "$REPO_DIR" && openenv validate 2>&1) && VALIDATE_OK=true
if [ "$VALIDATE_OK" = true ]; then
  pass "openenv validate passed"
  [ -n "$VALIDATE_OUTPUT" ] && log "  $VALIDATE_OUTPUT"
else
  fail "openenv validate failed"
  printf "%s\n" "$VALIDATE_OUTPUT"
  stop_at "Step 3"
fi

printf "\n${BOLD}========================================${NC}\n"
printf "${GREEN}${BOLD}  All 3/3 checks passed! Ready to submit.${NC}\n"
printf "${BOLD}========================================${NC}\n\n"
exit 0
