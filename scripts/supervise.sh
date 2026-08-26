#!/usr/bin/env bash
# Keep the live runner alive for a long unattended session.
#
# The runner is the thing that trades; this only restarts it. Two guard rails
# matter here:
#
#   * The runner takes an flock (data/state/<mode>.lock) and refuses to start
#     if another instance holds it. Two runners with different universes once
#     fought each other -- each cycle one bought what the other had just sold,
#     costing 0.72% in fees with no net position change. If the lock is held,
#     this exits rather than adding a second trader.
#   * Backoff on repeated fast failures, so a permanent error (bad credentials,
#     a delisted market) does not become a hot restart loop against the API.
set -uo pipefail
cd "$(dirname "$0")/.."

DEADLINE=$(( $(date +%s) + ${RUN_SECONDS:-345600} ))   # default 4 days
LOG_DIR=data/logs
mkdir -p "$LOG_DIR"

fails=0
while [ "$(date +%s)" -lt "$DEADLINE" ]; do
  remaining_min=$(( (DEADLINE - $(date +%s)) / 60 ))
  [ "$remaining_min" -lt 2 ] && break

  echo "[supervisor] $(date '+%F %T') starting runner, ${remaining_min}m left" \
    >> "$LOG_DIR/supervisor.log"

  started=$(date +%s)
  NBTREND_UNIVERSE=${NBTREND_UNIVERSE:-config/universe.live.yaml} \
    .venv/bin/python -m nbtrend.cli run \
      --yes --minutes "$remaining_min" --interval "${INTERVAL:-120}" \
      >> "$LOG_DIR/live.log" 2>&1
  code=$?
  ran=$(( $(date +%s) - started ))

  if [ "$code" -eq 0 ]; then
    echo "[supervisor] $(date '+%F %T') runner finished cleanly" >> "$LOG_DIR/supervisor.log"
    break
  fi

  # Exit code 1 with a near-instant exit is how the lock refusal looks.
  if [ "$ran" -lt 20 ]; then
    fails=$(( fails + 1 ))
  else
    fails=1
  fi

  if [ "$fails" -ge 5 ]; then
    echo "[supervisor] $(date '+%F %T') 5 fast failures (last code $code); giving up" \
      >> "$LOG_DIR/supervisor.log"
    break
  fi

  backoff=$(( fails * 30 ))
  echo "[supervisor] $(date '+%F %T') runner exited $code after ${ran}s; retry in ${backoff}s" \
    >> "$LOG_DIR/supervisor.log"
  sleep "$backoff"
done

echo "[supervisor] $(date '+%F %T') supervisor exiting" >> "$LOG_DIR/supervisor.log"
