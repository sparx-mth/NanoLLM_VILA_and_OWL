#!/bin/bash
# Configuration for the remote Jetson environment
IMAGE_NAME="nanoowl_new:v1.5"
MOUNT_PATH="/home/user/nano_owl_tree"
CONTAINER_NAME="now_eng"

# Ensure any existing stopped container with the same name is removed
docker rm -f $CONTAINER_NAME >/dev/null 2>&1

echo "Starting NanoOWL container: $CONTAINER_NAME"
docker run -d --name $CONTAINER_NAME \
  --runtime nvidia \
  --network host --ipc=host \
  -v $MOUNT_PATH:/app/nano_owl_tree \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  $IMAGE_NAME \
  /bin/bash -c "pip3 install -e /app/nano_owl_tree/nano_owl_tree/ && nanoowl-service --engine /opt/nanoowl/data/owl_image_encoder_patch32.engine"