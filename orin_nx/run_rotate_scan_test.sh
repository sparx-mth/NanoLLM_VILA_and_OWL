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
EXAMPLES_DIR="/home/iai/iai_ws/src/examples/src"
# Host-side path of the iai container's /home/iai/frames mount (where rotate_360_example.py
# actually writes - /home/iai/captures isn't mounted into that container). The pipeline
# and display server work in the per-drone R<id> subfolder next to the raw scans.
FRAMES_HOST_ROOT="${HOME}/iai_ws/frames"
CAPTURES_ROOT="${FRAMES_HOST_ROOT}/R${DRONE_ID}"

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

echo "[pipeline] copying the rotation scan into ${CAPTURES_ROOT}"
latest_scan=$(ls -dt "${FRAMES_HOST_ROOT}"/rotate_scan_*/ 2>/dev/null | head -1)
if [ -z "$latest_scan" ]; then
  echo "[pipeline] no rotate_scan_* folder found under ${FRAMES_HOST_ROOT} -- aborting" >&2
  exit 1
fi
mkdir -p "${CAPTURES_ROOT}"
scan_name=$(basename "${latest_scan%/}")
cp -r "$latest_scan" "${CAPTURES_ROOT}/${scan_name}"
ln -sfn "${CAPTURES_ROOT}/${scan_name}" "${CAPTURES_ROOT}/latest"
echo "[pipeline] latest -> ${CAPTURES_ROOT}/${scan_name}"

cd ~/NanoLLM_VILA_and_OWL
echo "[pipeline] starting comm_manager_vllm.py in the background (it never exits on its own)"
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
  --sleep-between 2 --force &
COMM_PID=$!
trap 'kill "$COMM_PID" 2>/dev/null' EXIT

echo "[pipeline] starting display server on :8090 (frames will appear as comm_manager finishes them)"
python3 display_server.py --root "${CAPTURES_ROOT}" --host 0.0.0.0 --port 8090 --latest-only
