# Wait for every worker before returning the last failure, if any.
# Callers use the + expansion to support an empty PID list under Bash 3 with set -u.
wait_for_jobs() {
  local pid failed=0
  for pid in "$@"; do
    wait "$pid" || {
      failed=$?
      printf 'Worker process %s exited with status %s\n' "$pid" "$failed" >&2
    }
  done
  return "$failed"
}
