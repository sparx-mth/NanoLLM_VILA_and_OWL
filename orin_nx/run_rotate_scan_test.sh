#!/bin/bash
# Pre/post-flight helper for the 360-rotate scan test (TheAgency repo's
# rotate_360_example.py). You run the flight itself (rotate_360_example.py, then
# land_and_disarm_example.py) yourself, in your own terminal, under your own control -
# this script only checks the vision containers are ready beforehand, and runs the
# vision pipeline on the captured frames afterward, once you confirm you've landed.
#
# Run from the host (not inside any container).
set -u

DRONE_ID=$(grep '^ROS_DOMAIN_ID=' ~/iai_ws/src/docker/.env | cut -d= -f2)
CAPTURES_ROOT="/home/iai/captures/R${DRONE_ID}"
EXAMPLES_DIR="/home/iai/iai_ws/src/examples/src"

check_answering() {  # check_answering <port> <label>
  local port="$1" label="$2"
  if curl -s -o /dev/null "http://127.0.0.1:${port}/"; then
    echo "[check] ${label} (:${port}) answering"
  else
    echo "[check] ${label} (:${port}) NOT answering -- fix this before flying" >&2
    exit 1
  fi
}

echo "[check] verifying vLLM, NanoOWL and DA3 are up and answering..."
check_answering 8080 "vLLM"
check_answering 5060 "NanoOWL"
check_answering 5070 "DA3"
echo "[check] all three answering -- safe to fly"

echo
echo "Now run the flight yourself, in your own terminal:"
echo "  docker exec -it iai python3 ${EXAMPLES_DIR}/rotate_360_example.py"
echo "  (it starts capturing frames on its own once Position mode is confirmed)"
echo "  docker exec -it iai python3 ${EXAMPLES_DIR}/land_and_disarm_example.py"
echo
read -rp "Press Enter once you've landed and disarmed, to run the vision pipeline on the captured frames: "

echo "[pipeline] pointing 'latest' at the rotation scan folder"
latest_scan=$(ls -dt "${CAPTURES_ROOT}"/rotate_scan_*/ 2>/dev/null | head -1)
if [ -z "$latest_scan" ]; then
  echo "[pipeline] no rotate_scan_* folder found under ${CAPTURES_ROOT} -- aborting" >&2
  exit 1
fi
ln -sfn "$latest_scan" "${CAPTURES_ROOT}/latest"
echo "[pipeline] latest -> ${latest_scan}"

cd ~/NanoLLM_VILA_and_OWL
python3 comm_manager_vllm.py \
  --no-config \
  --host 0.0.0.0 --port 5050 \
  --captures-root "${CAPTURES_ROOT}" \
  --endpoint http://127.0.0.1:8080 \
  --nanoowl-endpoint http://127.0.0.1:5060/infer \
  --vllm-url http://127.0.0.1:8080 \
  --vllm-model Qwen/Qwen2-VL-2B-Instruct \
  --vllm-max-tokens 192 \
  --depth-endpoint http://127.0.0.1:5070/bbox_depth \
  --forward-json-url "" \
  --vlm-timeout 60 --retries 3 --retry-sleep 2 \
  --sleep-between 2 --force

echo "[pipeline] starting display server on :8090"
python3 display_server.py --root "${CAPTURES_ROOT}" --host 0.0.0.0 --port 8090 --latest-only
