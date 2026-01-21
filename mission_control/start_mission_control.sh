#!/bin/bash
# start_mission_control.sh

echo "Setting up Mission Control..."
cd "$(dirname "$0")"

if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    ./venv/bin/pip install flask requests
fi

# Start the local agent for the laptop node in background
echo "Starting local Node Agent (port 6000)..."
./venv/bin/python3 node_agent.py --port 6000 > agent.log 2>&1 &
AGENT_PID=$!

echo "Starting Mission Control Server..."
./venv/bin/python3 mission_control.py

# Cleanup when server exits
kill $AGENT_PID
