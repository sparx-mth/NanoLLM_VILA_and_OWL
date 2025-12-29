#!/bin/bash
# Configuration for VILA service
IMAGE_NAME="nano_llm_custom"
CONTAINER_NAME="vila_service"
DATA_VOLUME="/home/user/jetson-containers/data"
NOTIFY_URL="http://192.168.131.22:5050/from_vila"

# Remove existing container if it exists
docker rm -f $CONTAINER_NAME >/dev/null 2>&1

echo "Starting VILA container: $CONTAINER_NAME"

# Using jetson-containers wrapper to handle NVIDIA runtime and internal mounts
jetson-containers run -d \
  --name $CONTAINER_NAME \
  --publish 8080:8080 \
  --volume $DATA_VOLUME:$DATA_VOLUME \
  $IMAGE_NAME \
  /bin/bash -c "python3 -m nano_llm.chat \
    --api=mlc \
    --model Efficient-Large-Model/VILA1.5-3b \
    --max-context-len 256 \
    --max-new-tokens 32 \
    --save-json-by-image \
    --server \
    --port 8080 \
    --notify-url $NOTIFY_URL"

# Verification
sleep 3
if [ "$(docker ps -q -f name=$CONTAINER_NAME)" ]; then
    echo "VILA container is running successfully."
else
    echo "ERROR: VILA container failed to start. Check logs with: docker logs $CONTAINER_NAME"
fi
