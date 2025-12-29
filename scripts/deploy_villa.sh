#!/bin/bash
# Local deployment script for VILA service
cd "$(dirname "$0")/.."

JETSON_IP="192.168.131.22"
JETSON_USER="user"
REMOTE_PATH="/home/user/villa_service"
VILA_CONTAINER="vila_service"

echo "1. Syncing VILA run script to $JETSON_IP..."
ssh $JETSON_USER@$JETSON_IP "mkdir -p $REMOTE_PATH/scripts"
rsync -avz ./scripts/run_villa_container.sh $JETSON_USER@$JETSON_IP:$REMOTE_PATH/scripts/

echo "2. Checking if VILA container is running..."
RUNNING=$(ssh $JETSON_USER@$JETSON_IP "docker ps -q -f name=^/${VILA_CONTAINER}$")

if [ -z "$RUNNING" ]; then
    echo "   VILA container not found. Starting it now..."
    ssh $JETSON_USER@$JETSON_IP "chmod +x $REMOTE_PATH/scripts/run_villa_container.sh && $REMOTE_PATH/scripts/run_villa_container.sh"
else
    echo "   VILA container is already running."
fi

echo "VILA Deployment complete."