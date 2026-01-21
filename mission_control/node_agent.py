import flask
from flask import request, jsonify, Response
import subprocess
import threading
import time
import os
import signal
import queue
import logging

app = flask.Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Store managed processes: service_id -> Popen object
PROCESSES = {}
# Store output logs: service_id -> list of lines (maxlen=100)
LOGS = {}
MAX_LOG_LINES = 200

def enqueue_output(out, service_id, log_queue):
    for line in iter(out.readline, b''):
        line_str = line.decode('utf-8', errors='replace')
        log_queue.append(line_str)
        # Keep logs trimmed
        if len(log_queue) > MAX_LOG_LINES:
            log_queue.pop(0)
    out.close()

def start_service_process(service_id, command, cwd=None, env=None):
    if service_id in PROCESSES and PROCESSES[service_id].poll() is None:
        return False, "Service already running"

    try:
        logging.info(f"Starting {service_id}: {command} (cwd={cwd})")
        
        # Prepare environment
        run_env = os.environ.copy()
        if env:
            run_env.update(env)

        # Use shell=True to handle complex commands but be careful with security
        # Since this is an internal tool for a specific user content, it's acceptable.
        p = subprocess.Popen(
            command,
            shell=True,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, # Merge stderr into stdout
            env=run_env,
            preexec_fn=os.setsid # Create new process group
        )
        PROCESSES[service_id] = p
        
        if service_id not in LOGS:
            LOGS[service_id] = []
        
        # Start a thread to consume stdout
        t = threading.Thread(target=enqueue_output, args=(p.stdout, service_id, LOGS[service_id]))
        t.daemon = True
        t.start()
        
        return True, f"Started PID {p.pid}"
    except Exception as e:
        return False, str(e)

def stop_service_process(service_id):
    if service_id not in PROCESSES:
        return False, "Service not found"
    
    p = PROCESSES[service_id]
    if p.poll() is not None:
        del PROCESSES[service_id]
        return True, "Service was already stopped"

    try:
        # Kill the process group to ensure children die too (like shell wrapper)
        os.killpg(os.getpgid(p.pid), signal.SIGTERM)
        try:
            p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
            p.wait()
        
        del PROCESSES[service_id]
        return True, "Stopped"
    except Exception as e:
        return False, str(e)

@app.route('/start/<service_id>', methods=['POST'])
def handle_start(service_id):
    data = request.json or {}
    command = data.get('command')
    cwd = data.get('cwd')
    env = data.get('env')
    
    if not command:
        return jsonify({"ok": False, "error": "No command provided"}), 400
        
    ok, msg = start_service_process(service_id, command, cwd, env)
    return jsonify({"ok": ok, "message": msg})

@app.route('/stop/<service_id>', methods=['POST'])
def handle_stop(service_id):
    ok, msg = stop_service_process(service_id)
    return jsonify({"ok": ok, "message": msg})

@app.route('/status', methods=['GET'])
def handle_status():
    status_map = {}
    for sid, p in list(PROCESSES.items()):
        ret = p.poll()
        if ret is None:
            status_map[sid] = {"state": "running", "pid": p.pid}
        else:
            status_map[sid] = {"state": "stopped", "exit_code": ret}
            # Clean up stopped processes from dict? Maybe keep them to show exit code.
            # For now, keep them until restart.
    return jsonify({"services": status_map})

@app.route('/logs/<service_id>', methods=['GET'])
def handle_logs(service_id):
    if service_id not in LOGS:
        return jsonify({"lines": []})
    return jsonify({"lines": LOGS[service_id]})

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--port", type=int, default=6000)
    args = parser.parse_args()
    
    app.run(host='0.0.0.0', port=args.port)
