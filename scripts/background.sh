#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
phase="${1:?Specify a phase, e.g. smoke or generate}"
task="${2:-gsm8k}"
split="${3:-test}"
logs="${DATA_ROOT:-/mnt/data/wxh/wdl}/logs/StepCluster"
mkdir -p "$logs"
stamp="$(date +%Y%m%d-%H%M%S)"
log="$logs/${RUN_NAME:-qwen25-v1}-$phase-$task-$split-$stamp.log"
pidfile="$log.pid"
# Separate process group permits stopping only this job and its descendants.
nohup setsid bash scripts/run.sh "$phase" "$task" "$split" > "$log" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "$pid" > "$pidfile"
echo "PID/PGID $pid"
echo "Log $log"
echo "Follow: tail -n 30 -f '$log'"
echo "Stop after checking ps: kill -TERM -- -$pid"
