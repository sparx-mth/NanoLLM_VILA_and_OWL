#!/bin/bash
# install_agent.sh
# Usage: ./install_agent.sh --port 6000

PORT=6000
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --port) PORT="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

echo "Installing Node Agent dependencies..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

./venv/bin/pip install flask requests

echo "Starting Node Agent on port $PORT..."
./venv/bin/python3 node_agent.py --port $PORT
