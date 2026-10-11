#!/bin/bash
# Usage: ./run_da3.sh <path/to/model.onnx | model.engine> [extra server args...]
# The ONNX must sit next to its .onnx.data file. If given an .onnx the node builds the
# TensorRT engine itself on first start (slow, minutes) - engines are device-specific,
# so never reuse an engine built on the AGX.
set -e
MODEL=$(readlink -f "$1"); shift
[ -f "$MODEL" ] || { echo "model not found: $MODEL"; exit 1; }
MDIR=$(dirname "$MODEL"); MFILE=$(basename "$MODEL")
mkdir -p ~/da3_out

docker rm -f da3 2>/dev/null || true
docker run -d --restart unless-stopped --name da3 \
  --runtime nvidia --network host \
  -v "$MDIR":/models -v ~/da3_out:/out \
  da3:orin \
  ros2 run depth_anything_v3 depth_anything_http_server \
    --model /models/$MFILE \
    --camera-yaml /depth_anything_ws/src/ros2-depth-anything-v3-trt/camera_info_4k.yaml \
    --max-depth 7.0 --host 0.0.0.0 --port 5070 --save-depth \
    --tiling 1 --tile-overlap-div 2 --tile-fade-px 400 "$@"
