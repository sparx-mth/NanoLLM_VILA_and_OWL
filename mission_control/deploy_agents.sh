#!/bin/bash
# deploy_agents.sh
# Deploys mission control agent to remote nodes (agx1, nano, agx2)

# Define nodes: user@ip
# derived from system_config.json analysis
NODES=(
    "user@192.168.131.22"     # agx1
    "user@192.168.131.21"     # nano
    "nvidia@192.168.131.23"   # agx2
)

echo "=== Mission Control Agent Deployment ==="
echo "This script will copy the agent code to remote nodes and start it."
echo "You may be prompted for SSH passwords for each node."
echo ""

for NODE in "${NODES[@]}"; do
    echo "--------------------------------------------------"
    echo "Deploying to $NODE..."
    
    # 1. Create directory
    echo "Creating directory ~/mission_control_agent..."
    ssh $NODE "mkdir -p ~/mission_control_agent"
    if [ $? -ne 0 ]; then
        echo "Failed to connect or create directory on $NODE. Skipping."
        continue
    fi
    
    # 2. Copy files
    echo "Copying agent files..."
    scp node_agent.py install_agent.sh $NODE:~/mission_control_agent/
    
    # 3. Install and Start
    # uses install_agent.sh which creates venv and starts python script
    # we run it with nohup so it survives exit
    echo "Starting agent..."
    ssh $NODE "cd ~/mission_control_agent && chmod +x install_agent.sh && nohup ./install_agent.sh > agent.log 2>&1 &"
    
    echo "Deployment to $NODE triggered."
done

echo "--------------------------------------------------"
echo "Deployment loop finished."
echo "Please check Mission Control dashboard to see if nodes come online."
