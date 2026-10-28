#!/bin/bash
# Run on the LAPTOP (not the Jetson): opens one tmux window with 4 panes for a scan test.
#   1 takeoff  - shell inside the iai container, rotate_360_example.py typed but NOT run
#   2 land     - shell inside the iai container, land_and_disarm_example.py typed but NOT run
#   3 pipeline - run_rotate_scan_test.sh on the Jetson (checks services, waits for you)
#   4 tunnel   - forwards the Jetson display server to http://localhost:8090
# Each pane's output is also saved to ~/drone_logs/<session>_<date>/ on the laptop (ANSI
# colour codes stripped), so it survives closing the tmux session.
# Usage: drone_tmux.sh [n]   n = drone number (default 1) -> ssh host orin<n> from ~/.ssh/config
# Needs ssh keys (ssh-copy-id) or you type the password into the pane when it asks.
set -u

N="${1:-1}"
HOST="orin${N}"
SESSION="drone${N}"
EXAMPLES="/home/iai/iai_ws/src/examples/src"
WAIT_S=90

if tmux has-session -t "$SESSION" 2>/dev/null; then
  exec tmux attach -t "$SESSION"
fi

# wait_prompt <pane>: block until the pane's last line looks like the Jetson/container
# shell prompt (so typed text isn't swallowed by a password prompt or the login banner).
wait_prompt() {
  local pane="$1" i
  for ((i = 0; i < WAIT_S; i++)); do
    if tmux capture-pane -p -t "$pane" | grep -v '^[[:space:]]*$' | tail -1 \
         | grep -qE 'iai@[^ ]+:[^ ]*\$ *$'; then
      return 0
    fi
    sleep 1
  done
  echo "pane $pane: no prompt after ${WAIT_S}s - typed nothing there" >&2
  return 1
}

tmux new-session -d -s "$SESSION" -x 220 -y 55 -n scan
tmux set-option -t "$SESSION" mouse on   # click a pane to focus it, drag borders, scroll
tmux set-option -t "$SESSION" pane-border-status top
tmux set-option -t "$SESSION" pane-border-format ' #{pane_title} '

P1=$(tmux display-message -p -t "$SESSION:scan.0" '#{pane_id}')
P2=$(tmux split-window -h -t "$P1" -P -F '#{pane_id}')
P3=$(tmux split-window -v -t "$P1" -P -F '#{pane_id}')
P4=$(tmux split-window -v -t "$P2" -P -F '#{pane_id}')
tmux select-layout -t "$SESSION:scan" tiled

tmux select-pane -t "$P1" -T "1 TAKEOFF (press Enter to run)"
tmux select-pane -t "$P2" -T "2 LAND + DISARM (press Enter to run)"
tmux select-pane -t "$P3" -T "3 PIPELINE"
tmux select-pane -t "$P4" -T "4 DISPLAY TUNNEL -> http://localhost:8090"

LOG_DIR="${HOME}/drone_logs/${SESSION}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
log_pane() {  # log_pane <pane_id> <name>
  tmux pipe-pane -t "$1" -o "sed -u -e 's/\x1b\[[0-9;?]*[a-zA-Z]//g' -e 's/\r//g' >> '${LOG_DIR}/$2.log'"
}
log_pane "$P1" "1_takeoff"
log_pane "$P2" "2_land"
log_pane "$P3" "3_pipeline"
log_pane "$P4" "4_tunnel"

# Pipeline and tunnel just run; the container panes open a shell, then get the command typed.
tmux send-keys -t "$P3" "ssh -t $HOST 'cd ~/NanoLLM_VILA_and_OWL/orin_nx && ./run_rotate_scan_test.sh; exec bash'" Enter
tmux send-keys -t "$P4" "ssh -N -o ExitOnForwardFailure=yes -L 8090:localhost:8090 $HOST" Enter
tmux send-keys -t "$P1" "ssh -t $HOST 'docker exec -it iai bash'" Enter
tmux send-keys -t "$P2" "ssh -t $HOST 'docker exec -it iai bash'" Enter

# Type the commands in the background so you're attached immediately and can answer any
# ssh password prompt yourself; each command appears once its pane reaches a shell prompt.
( wait_prompt "$P1" && tmux send-keys -t "$P1" "python3 $EXAMPLES/rotate_360_example.py" ) &
( wait_prompt "$P2" && tmux send-keys -t "$P2" "python3 $EXAMPLES/land_and_disarm_example.py" ) &

tmux select-pane -t "$P1"
exec tmux attach -t "$SESSION"
