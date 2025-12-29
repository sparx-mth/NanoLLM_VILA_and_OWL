#!/bin/bash
# Local deployment script to be run on your dev machine

# Ensure we are running from the project root regardless of where the script is called
cd "$(dirname "$0")/.." || exit

JETSON_IP="192.168.131.22"
JETSON_USER="user"
REMOTE_PATH="/home/user/nano_owl_tree"
CONTAINER_NAME="now_eng"

echo "1. Syncing selected files to $JETSON_IP..."
# Create the remote directory if it doesn't exist
ssh $JETSON_USER@$JETSON_IP "mkdir -p $REMOTE_PATH/scripts"

# Sync only the nano_owl_tree package and the run script
rsync -avz --exclude '__pycache__' \
      ./nano_owl_tree \
      $JETSON_USER@$JETSON_IP:$REMOTE_PATH/

rsync -avz \
      ./scripts/run_nano_owl_container_dev.sh \
      $JETSON_USER@$JETSON_IP:$REMOTE_PATH/scripts/

echo "2. Checking if container '$CONTAINER_NAME' is running..."
RUNNING=$(ssh $JETSON_USER@$JETSON_IP "docker ps -q -f name=^/${CONTAINER_NAME}$")

if [ -z "$RUNNING" ]; then
    echo "   Container not found. Starting it now..."
    ssh $JETSON_USER@$JETSON_IP "chmod +x $REMOTE_PATH/scripts/run_nano_owl_container_dev.sh && $REMOTE_PATH/scripts/run_nano_owl_container_dev.sh"
else
    echo "   Container is already running."
    echo "   Flask auto-reload is active if debug=True is set in nanoowl_service.py"
fi

echo "Deployment complete."