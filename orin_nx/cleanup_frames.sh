#!/bin/bash
# Prune old capture runs under a --captures-root so the disk doesn't fill.
#
# Retention is COUNT-based (keep the N most recent run folders), not age-based.
# This board has no clock source on the drone network and the time resets at
# boot (see README section 7) — an age-based rule ("delete anything older than
# N hours") would delete freshly captured frames the moment the clock resets
# to an epoch value after a reboot, since they'd suddenly look decades old.
# Counting folders is immune to that: it only cares about relative order.
#
# A run folder is named YYYY_MM_DD___HH_MM_SS (comm_manager_vllm.py / the raw
# capture script). Its paired <run>_ann and <run>_depth folders are pruned
# alongside it. The folder "latest" currently resolves to is never deleted,
# regardless of how old/new it sorts, since it's always in active use.
set -u

ROOT="${1:?usage: cleanup_frames.sh <captures-root> [keep-count]}"
KEEP="${2:-20}"

if [ ! -d "$ROOT" ]; then
  echo "[cleanup] root not found: $ROOT" >&2
  exit 1
fi

latest_target=""
if [ -L "$ROOT/latest" ]; then
  latest_target=$(readlink -f "$ROOT/latest")
fi

# Base run folders only: YYYY_MM_DD___HH_MM_SS, excluding their _ann/_depth siblings.
mapfile -t runs < <(find "$ROOT" -maxdepth 1 -mindepth 1 -type d \
  -regextype posix-extended -regex '.*/[0-9]{4}_[0-9]{2}_[0-9]{2}___[0-9]{2}_[0-9]{2}_[0-9]{2}' \
  | sort)

count=${#runs[@]}
if [ "$count" -le "$KEEP" ]; then
  echo "[cleanup] $count run(s) under $ROOT, keeping all (limit $KEEP)"
  exit 0
fi

to_delete_count=$((count - KEEP))
deleted=0

for run in "${runs[@]:0:$to_delete_count}"; do
  resolved=$(readlink -f "$run")
  if [ -n "$latest_target" ] && [ "$resolved" = "$latest_target" ]; then
    echo "[cleanup] skip (is 'latest'): $run"
    continue
  fi

  echo "[cleanup] removing: $run (+ _ann/_depth if present)"
  rm -rf -- "$run" "${run}_ann" "${run}_depth"
  deleted=$((deleted + 1))
done

echo "[cleanup] done: removed $deleted of $to_delete_count candidate run(s), $KEEP kept under $ROOT"
