from flask import Flask, render_template, jsonify, request
import requests
import json
import os
import concurrent.futures
import time

app = Flask(__name__)

# Load config
CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'system_config.json')
with open(CONFIG_PATH, 'r') as f:
    SYSTEM_CONFIG = json.load(f)

NODES = SYSTEM_CONFIG['nodes']
SERVICES = {s['id']: s for s in SYSTEM_CONFIG['services']}

def get_agent_url(node_name):
    if node_name not in NODES:
        return None
    node = NODES[node_name]
    return f"http://{node['ip']}:{node['agent_port']}"

@app.route('/')
def index():
    # Pass config to template for rendering structure
    return render_template('dashboard.html', config=SYSTEM_CONFIG)

def check_service_status(service):
    sid = service['id']
    node_name = service['node']
    base_url = get_agent_url(node_name)
    
    if not base_url:
        return sid, {"state": "error", "error": "Unknown node"}

    try:
        # We fetch full status of the node to minimize requests? 
        # Actually agent has /status for all its services.
        # But here we are doing per service query logic or bulk?
        # Let's just hit the agent's /status endpoint once per node.
        pass 
    except:
        pass
    return sid, {"state": "unknown"}

@app.route('/api/status')
def api_status():
    # 1. Identify unique nodes involved
    unique_nodes = list(NODES.keys())
    
    node_statuses = {}
    
    def fetch_node_status(node_name):
        url = get_agent_url(node_name)
        try:
            r = requests.get(f"{url}/status", timeout=1.5)
            if r.status_code == 200:
                data = r.json()
                return node_name, data.get("services", {}), None
            else:
                return node_name, {}, f"HTTP {r.status_code}"
        except Exception as e:
            return node_name, {}, "Unreachable"

    # Parallel fetch
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(unique_nodes)) as executor:
        future_to_node = {executor.submit(fetch_node_status, n): n for n in unique_nodes}
        for future in concurrent.futures.as_completed(future_to_node):
            node_name, services_status, error = future.result()
            node_statuses[node_name] = {"services": services_status, "error": error}

    # 2. Map back to our service list
    response = []
    for s in SYSTEM_CONFIG['services']:
        sid = s['id']
        node = s['node']
        
        node_data = node_statuses.get(node, {})
        node_error = node_data.get("error")
        service_state = node_data.get("services", {}).get(sid)
        
        status_entry = {
            "id": sid,
            "node": node,
            "state": "unknown",
            "metadata": {}
        }
        
        if node_error:
            status_entry["state"] = "offline"
            status_entry["message"] = f"Node {node_error}"
        elif service_state:
            status_entry["state"] = service_state["state"]
            status_entry["metadata"] = service_state
        else:
            status_entry["state"] = "stopped" # If agent is up but doesn't know about service, it's stopped
            
        response.append(status_entry)

    return jsonify({"services": response})

@app.route('/api/control/<service_id>/<action>', methods=['POST'])
def api_control(service_id, action):
    if service_id not in SERVICES:
        return jsonify({"ok": False, "error": "Unknown service"}), 404
        
    service = SERVICES[service_id]
    node_name = service['node']
    url = get_agent_url(node_name)
    
    if action == 'start':
        payload = {
            "command": service['command'],
            "cwd": service.get('cwd'),
            "env": service.get('env')
        }
        try:
            r = requests.post(f"{url}/start/{service_id}", json=payload, timeout=2)
            return jsonify(r.json()), r.status_code
        except Exception as e:
             return jsonify({"ok": False, "error": str(e)}), 500
             
    elif action == 'stop':
         try:
            r = requests.post(f"{url}/stop/{service_id}", timeout=2)
            return jsonify(r.json()), r.status_code
         except Exception as e:
             return jsonify({"ok": False, "error": str(e)}), 500
    
    return jsonify({"ok": False, "error": "Invalid action"}), 400

@app.route('/api/logs/<service_id>')
def api_logs(service_id):
    if service_id not in SERVICES:
        return jsonify({"lines": ["Unknown service"]})
    
    service = SERVICES[service_id]
    node_name = service['node']
    url = get_agent_url(node_name)
    
    try:
        r = requests.get(f"{url}/logs/{service_id}", timeout=2)
        return jsonify(r.json())
    except Exception as e:
        return jsonify({"lines": [f"Error fetching logs: {e}"]})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
