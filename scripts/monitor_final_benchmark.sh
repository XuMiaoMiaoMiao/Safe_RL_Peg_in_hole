#!/usr/bin/env bash
# Lightweight overnight monitor for scripts/run_final_benchmark_2026-05-24.sh.
# It records progress snapshots and produces a final markdown report after the
# benchmark process exits. It is intentionally read-only.

set -u

LOGDIR="${LOGDIR:-/tmp/final_bench_logs}"
SUMMARY="${SUMMARY:-/tmp/final_bench_summary.txt}"
INTERVAL="${INTERVAL:-60}"
STAMP="${STAMP:-$(date +%Y%m%d)}"
MONITOR_LOG="${MONITOR_LOG:-/tmp/final_bench_monitor_${STAMP}.log}"
REPORT="${REPORT:-/tmp/final_bench_monitor_report_${STAMP}.md}"
ERROR_RE='Traceback|RuntimeError|Error:|Exception|TIMEOUT|timeout|FAILED|failed|CUDA error|HydraEngine|Segmentation fault'

now() {
    date '+%Y-%m-%d %H:%M:%S'
}

latest_log() {
    ls -t "$LOGDIR"/*.log 2>/dev/null | head -1
}

has_runner() {
    pgrep -f 'run_final_benchmark_2026-05-24.sh|train_sac_lagrangian.py|train_sac.py' >/dev/null 2>&1
}

append_snapshot() {
    local latest
    latest="$(latest_log || true)"

    {
        echo
        echo "================================================================"
        echo "[$(now)] monitor snapshot"
        echo "================================================================"
        echo
        echo "-- processes --"
        pgrep -af 'run_final_benchmark_2026-05-24.sh|train_sac_lagrangian.py|train_sac.py' || true
        echo
        echo "-- summary tail --"
        tail -40 "$SUMMARY" 2>/dev/null || true
        echo
        echo "-- latest log --"
        if [[ -n "${latest}" ]]; then
            echo "$latest"
            echo
            echo "last epochs:"
            grep -E 'Epoch [0-9]+ \|' "$latest" 2>/dev/null | tail -5 || true
            echo
            echo "last cost lines:"
            grep -E 'rollout_ep_cost|epoch_collision' "$latest" 2>/dev/null | tail -5 || true
            echo
            echo "recent errors in latest log:"
            grep -E "$ERROR_RE" "$latest" 2>/dev/null | tail -20 || true
        else
            echo "(no logs yet)"
        fi
    } >> "$MONITOR_LOG"
}

write_report() {
    {
        echo "# Final Benchmark Overnight Monitor Report"
        echo
        echo "- Generated: $(now)"
        echo "- Log dir: \`$LOGDIR\`"
        echo "- Summary: \`$SUMMARY\`"
        echo "- Monitor log: \`$MONITOR_LOG\`"
        echo
        echo "## Process State"
        echo
        if has_runner; then
            echo "Benchmark processes still running:"
            echo
            echo '```text'
            pgrep -af 'run_final_benchmark_2026-05-24.sh|train_sac_lagrangian.py|train_sac.py' || true
            echo '```'
        else
            echo "No benchmark training process is running."
        fi
        echo
        echo "## Summary Tail"
        echo
        echo '```text'
        tail -120 "$SUMMARY" 2>/dev/null || true
        echo '```'
        echo
        echo "## Per-Log Progress"
        echo
        echo '```text'
        for f in "$LOGDIR"/*.log; do
            [[ -e "$f" ]] || continue
            last_epoch="$(grep -E 'Epoch [0-9]+ \|' "$f" 2>/dev/null | tail -1 || true)"
            last_cost="$(grep -E 'rollout_ep_cost|epoch_collision' "$f" 2>/dev/null | tail -1 || true)"
            printf '%s\n' "$f"
            printf '  %s\n' "${last_epoch:-no epoch line}"
            printf '  %s\n' "${last_cost:-no cost line}"
        done
        echo '```'
        echo
        echo "## Errors / Timeouts"
        echo
        echo '```text'
        grep -R -n -E "$ERROR_RE" "$LOGDIR" "$SUMMARY" 2>/dev/null || true
        echo '```'
    } > "$REPORT"
}

echo "[$(now)] monitor started" >> "$MONITOR_LOG"
append_snapshot

while has_runner; do
    sleep "$INTERVAL"
    append_snapshot
done

echo "[$(now)] benchmark process ended; writing report" >> "$MONITOR_LOG"
append_snapshot
write_report
echo "[$(now)] report written to $REPORT" >> "$MONITOR_LOG"
