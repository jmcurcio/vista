#!/usr/bin/env bash
# Follow a VISTA Claude/oMLX run as readable events: what the model says,
# which tools it calls and what each returns (text only; images are in
# artifacts/evidence), compactions, retries and final answers.
#
#   ./scripts/watch_run.sh RUN_DIR [PID]
#
# Waits for the run to start, switches to each new stream as it appears (a
# recovery segment, or the next question of a batch), and exits when PID ends.
# Without PID it follows until Ctrl-C. To launch a run and watch it in one
# terminal:
#
#   uv run vista ... --output runs/NAME & ./scripts/watch_run.sh runs/NAME $!
#
# Tool results longer than WATCH_RESULT_CHARS (default 1500) are cut; set it to
# 0 to show them in full.
set -uo pipefail
run=${1:?usage: watch_run.sh RUN_DIR [PID]}
pid=${2:-}
limit=${WATCH_RESULT_CHARS:-1500}
filter='
  def clip: if $limit > 0 and length > $limit
    then .[0:$limit] + " …(\(length - $limit) more chars)" else . end;
  if .type=="assistant" then .message.content[] |
     if .type=="text" then "SAYS: " + (.text|gsub("\n";" "))
     elif .type=="tool_use" then "CALL " + (.name|sub("mcp__game__";"")) + ": " + (.input.question // "")
     elif .type=="thinking" then "(thought \(.thinking|length) chars)" else empty end
  elif .type=="user" then .message.content[]? | select(.type?=="tool_result")
     | "  -> result" + (if .is_error then " ERROR" else "" end) + ": "
       + ((if (.content|type)=="string" then .content
           else [.content[]? | select(.type=="text") | .text] | join(" ") end)
          | gsub("\n";" ") | clip)
  elif .type=="system" and .subtype=="compact_boundary"
     then "** COMPACTED \(.compact_metadata.pre_tokens) -> \(.compact_metadata.post_tokens) tokens"
  elif .type=="system" and .subtype=="api_retry" then "!! API retry (\(.error))"
  elif .type=="result" then "END (\(.terminal_reason // .subtype)): \(.result)"
  else empty end'

follower=
current=
stop() {
  [[ -n $follower ]] || return 0
  pkill -P "$follower" 2>/dev/null
  kill "$follower" 2>/dev/null
  wait "$follower" 2>/dev/null
}
trap 'stop; exit' INT TERM
while :; do
  newest=$(find "$run" -name 'segment_*.stdout.jsonl' -not -path '*/preflight/*' \
    -print0 2>/dev/null | xargs -0 ls -t 2>/dev/null | head -1)
  if [[ -n $newest && $newest != "$current" ]]; then
    stop
    current=$newest
    printf '\n== %s\n' "${current#"$run"/}"
    (tail -n +1 -F "$current" \
      | jq -r --unbuffered --argjson limit "$limit" "$filter") 2>/dev/null &
    follower=$!
  elif [[ -z $current ]]; then
    printf '\rWaiting for %s to start...' "$run"
  fi
  if [[ -n $pid ]] && ! kill -0 "$pid" 2>/dev/null; then
    sleep 2  # let the last lines through
    stop
    printf '\n== run process %s ended\n' "$pid"
    exit 0
  fi
  sleep 2
done
