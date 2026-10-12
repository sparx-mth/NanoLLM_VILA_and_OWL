#!/bin/bash
# Bring up vLLM -> NanoOWL -> DA3 -> iai in strict order. They share the Orin NX's 15.6 GB
# of memory, and vLLM sizes its GPU budget at startup, so anything that loads while vLLM is
# still starting can make it fail ("Free memory ... less than desired" / "No available
# memory for the cache blocks"). Run as root (drops the page cache, which CUDA doesn't
# count as free on Jetson).
#
# iai (the ROS2 flight-controller/camera bridge, managed by its own iai.service, not
# plain `docker run`) is light on its own but was seen crashing vLLM when both started at
# the same moment (free memory only 7.99/15.29 GiB, below vLLM's ~8.4 GiB startup need).
# It isn't actually a conflict once vLLM has already claimed its budget - only a race if
# they start together, which is why it's last here and disabled from its own independent
# auto-start (`sudo systemctl disable iai.service` - this script starts it explicitly).
set -u

wait_http() {  # wait_http <port> <seconds> : any HTTP answer means the server is up
  local port=$1 deadline=$((SECONDS + $2))
  until curl -s -o /dev/null "http://127.0.0.1:${port}/"; do
    [ $SECONDS -ge $deadline ] && { echo "timeout waiting for port $port"; return 1; }
    sleep 5
  done
}

echo "[start] stopping DA3, NanoOWL and iai so vLLM starts alone"
docker stop da3 now_eng >/dev/null 2>&1 || true
systemctl stop iai.service >/dev/null 2>&1 || true

sync && echo 3 > /proc/sys/vm/drop_caches

echo "[start] vLLM"
docker start vllm_qwen >/dev/null
wait_http 8080 900 || exit 1

echo "[start] NanoOWL"
docker start now_eng >/dev/null
wait_http 5060 300 || exit 1

echo "[start] DA3"
docker start da3 >/dev/null
wait_http 5070 120 || exit 1

echo "[start] iai (ROS2 bridge - no HTTP port, just launch it; its own Restart=always
and hardware/RPi wait loop take it from here)"
systemctl start iai.service >/dev/null 2>&1 || true

echo "[start] all services up"
