#!/system/bin/sh
set -eu

./qnn_executor_runner \
  --model_path=spatial_sr_qnn_htp_fp16.pte \
  --input_list_path=missing_input_list.txt \
  --warm_up=5 \
  --iteration=5000 \
  --etdump_path=memory_etdump.etdp \
  >memory_runner.log 2>&1 &
runner_pid=$!
peak_rss_kb=0
while kill -0 "$runner_pid" 2>/dev/null; do
  current_rss_kb="$(awk '/^VmRSS:/ {print $2}' "/proc/$runner_pid/status" 2>/dev/null || true)"
  if [ -n "$current_rss_kb" ] && [ "$current_rss_kb" -gt "$peak_rss_kb" ]; then
    peak_rss_kb="$current_rss_kb"
  fi
  sleep 0.01
done
wait "$runner_pid"
printf '%s\n' "$peak_rss_kb" >memory_peak_rss_kb.txt
