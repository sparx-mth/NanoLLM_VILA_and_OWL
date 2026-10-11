#!/bin/bash
# Bring up vLLM -> NanoOWL -> DA3 in strict order. They share the Orin NX's 15.6 GB of
# memory, and vLLM sizes its GPU budget at startup, so anything that loads while vLLM is
# still starting can make it fail ("Free memory ... less than desired" / "No available
# memory for the cache blocks"). Run as root (drops the page cache, which CUDA doesn't
# count as free on Jetson).
set -u

wait_http() {  # wait_http <port> <seconds> : any HTTP answer means the server is up
  local port=$1 deadline=$((SECONDS + $2))
  until curl -s -o /dev/null "http://127.0.0.1:${port}/"; do
    [ $SECONDS -ge $deadline ] && { echo "timeout waiting for port $port"; return 1; }
    sleep 5
  done
}

echo "[start] stopping DA3 and NanoOWL so vLLM starts alone"
docker stop da3 now_eng >/dev/null 2>&1 || true

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

echo "[start] all services up"
