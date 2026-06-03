#!/usr/bin/env bash
# Aggregate Julia --code-coverage .cov files into per-file + global line coverage.
# Assumes a SINGLE coverage run is present (clean stale .cov first). Picks the
# dominant PID so a stray older .cov can't pollute the numbers.
# Usage: scripts/cov_aggregate.sh [src_dir]   (default: src)
set -euo pipefail
SRC="${1:-src}"
PID=$(find "$SRC" -name '*.cov' | sed -E 's/.*\.([0-9]+)\.cov/\1/' | sort | uniq -c | sort -rn | head -1 | awk '{print $2}')
[ -z "${PID:-}" ] && { echo "no .cov files under $SRC — run tests with coverage first"; exit 1; }
tmp=$(mktemp)
for f in $(find "$SRC" -name "*.$PID.cov"); do
  awk '{ if ($1 ~ /^[0-9]+$/) { tot++; if ($1+0>0) cov++ } }
       END { s=FILENAME; sub(/\.[0-9]+\.cov$/,"",s); if(tot>0) printf "%5.1f %6d %6d %s\n", 100*cov/tot, cov, tot, s }' "$f"
done | sort -n > "$tmp"
echo "=== per-file coverage (low -> high): pct cov tot file ==="
cat "$tmp"
echo "=== files by absolute uncovered lines ==="
awk '{printf "%5d uncov (%4.1f%%) %s\n", $3-$2, $1, $4}' "$tmp" | sort -rn | head -25
echo "=== GLOBAL ==="
awk '{c+=$2;t+=$3} END{printf "covered=%d total=%d pct=%.2f%%\n",c,t,100*c/t}' "$tmp"
rm -f "$tmp"
