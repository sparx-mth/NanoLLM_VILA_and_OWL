import streamlit as st
import subprocess
import os
import yaml

import shlex
# --- CONFIG ---
JETSON_PWD = "1"


def load_config():
    yaml_path = "config/networks.yaml"
    if not os.path.exists(yaml_path):
        st.error(f"Config file not found at {yaml_path}")
        return None
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)


config = load_config()
if not config:
    st.stop()

profiles = list(config['profiles'].keys())
default_profile = config['defaults']['profile']

# --- SIDEBAR ---
st.sidebar.header("Network Settings")
selected_profile = st.sidebar.selectbox("Active Profile", profiles, index=profiles.index(default_profile))
profile_data = config['profiles'][selected_profile]

use_tiling = st.sidebar.toggle("Enable Depth Tiling", value=False)
tiling_val = "1" if use_tiling else "0"


# --- REMOTE UTILITIES ---
def get_service_target(service_key):
    svc = profile_data['services'][service_key]
    host_alias = svc['host']
    host_info = profile_data['hosts'][host_alias]
    return {
        "user": host_info['user'],
        "ip": host_info['ip'],
        "port": svc['port']
    }

def get_log(target, lines=120):
    log_path = f"/tmp/{target['port']}.log"
    res = ssh_exec(
        target,
        f"test -f {log_path} && tail -n {lines} {log_path} || echo 'No log yet: {log_path}'",
        background=False,
    )
    return res.stdout if res.stdout else res.stderr


def clear_log(target):
    log_path = f"/tmp/{target['port']}.log"
    return ssh_exec(
        target,
        f"rm -f {log_path}",
        background=False,
    )

def ssh_exec(target, cmd, background=True, use_sudo=False):
    log_path = f"/tmp/{target['port']}.log"

    if background:
        wrapped_cmd = (
            f"rm -f {log_path}; "
            f"nohup bash -lc {shlex.quote(cmd)} > {log_path} 2>&1 &"
        )
        if use_sudo:
            wrapped_cmd = f"echo {shlex.quote(JETSON_PWD)} | sudo -S bash -lc {shlex.quote(wrapped_cmd)}"
    else:
        wrapped_cmd = cmd
        if use_sudo:
            wrapped_cmd = f"echo {shlex.quote(JETSON_PWD)} | sudo -S bash -lc {shlex.quote(cmd)}"

    ssh_cmd = [
        "sshpass", "-p", JETSON_PWD,
        "ssh",
        "-o", "StrictHostKeyChecking=no",
        f"{target['user']}@{target['ip']}",
        wrapped_cmd,
    ]

    if background:
        return subprocess.Popen(ssh_cmd)

    return subprocess.run(
        ssh_cmd,
        capture_output=True,
        text=True,
    )


def check_status(target):
    res = ssh_exec(
        target,
        f"lsof -nP -iTCP:{target['port']} -sTCP:LISTEN || true",
        background=False,
        use_sudo=True,
    )
    return len(res.stdout.strip()) > 0


# --- SERVICE DEFINITIONS ---
t_vllm = get_service_target("vila_api")
t_owl = get_service_target("nanoowl")
print(t_owl)
t_depth = get_service_target("depth")
t_comm = get_service_target("comm_manager")
t_disp = get_service_target("display_server")

SERVICES = {
    "vLLM Server": {
        "target": t_vllm,
        "container_name": "vllm_server",
        "description": "Vision-language API",
        "cmd": (
            "docker rm -f vllm_server >/dev/null 2>&1 || true; "
            "docker run -d --name vllm_server "
            "--runtime nvidia --network host "
            "-v ~/my_models/qwen3-vl-4b:/app/model "
            "-e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 "
            "vllm_qwen3_vl_4b_instruct_aws_4bit:latest "
            f"vllm serve /app/model --host {t_vllm['ip']} --port {t_vllm['port']} "
            "--dtype float16 --gpu-memory-utilization 0.5 --max-model-len 2048 --enforce-eager"
        ),
        "stop_cmd": "docker rm -f vllm_server >/dev/null 2>&1 || true",
    },
    "NanoOWL": {
        "target": t_owl,
        "cmd": (
            "docker rm -f nanoowl_service >/dev/null 2>&1 || true; "
            "docker run -d --name nanoowl_service "
            "--runtime nvidia "
            "--network host --ipc=host "
            "-e NVIDIA_VISIBLE_DEVICES=all "
            "-e NVIDIA_DRIVER_CAPABILITIES=all "
            "-e LD_LIBRARY_PATH=/usr/local/lib:/usr/lib/aarch64-linux-gnu:/usr/lib:/lib "
            "nanoowl_new:v1.5 "
            "python3 /opt/nanoowl/examples/jetson_server/nanoowl_service.py "
            "--engine /opt/nanoowl/data/owl_image_encoder_patch32.engine "
            f"--host {t_owl['ip']} --port {t_owl['port']} --min-score 0.2"
        ),
        "stop_cmd": "docker rm -f nanoowl_service >/dev/null 2>&1 || true",
    },
    "Depth V3": {
        "target": t_depth,
        "description": "DA3 depth HTTP server",
        "cmd": (
            "source /opt/ros/humble/setup.bash && "
            "source /home/user/depth_anything_ws/install/setup.bash && "
            "ros2 run depth_anything_v3 depth_anything_http_server "
            "--model /home/user/depth_anything_ws/src/ros2-depth-anything-v3-trt/onnx/DA3-SMALL/DA3-SMALL.fp16-batch1.engine "
            "--camera-yaml /home/user/GIT/TheAgency/sparx_agency/robots/XTEND/config/camera_xtend_ros_calib_720_420.yaml "
            "--max-depth 7.0 "
            "--save-depth "
            f"--host {t_depth['ip']} --port {t_depth['port']} --tiling {tiling_val}"
        ),
        "stop_cmd": f"fuser -k {t_depth['port']}/tcp || true",
    },
    "Comm Manager": {
        "target": t_comm,
        "cmd": (
            f"cd /home/user/GIT/NanoLLM_VILA_and_OWL && "
            f"python3 comm_manager_vllm.py "
            f"--profile {selected_profile} "
            f"--vllm-model espressor/meta-llama.Llama-3.2-3B-Instruct_W4A16 "
            f"--captures-root /home/user/jetson-containers/data/R1/ "
            f"--endpoint http://{t_vllm['ip']}:{t_vllm['port']} "
            f"--host {t_comm['ip']} "
            f"--force "
            f"--depth-endpoint http://{t_depth['ip']}:{t_depth['port']}/bbox_depth"
        ),
        "stop_cmd": f"fuser -k {t_comm['port']}/tcp || true",
    },
    "Data Server": {
        "target": {"user": t_vllm["user"], "ip": t_vllm["ip"], "port": 9000},
        "description": "Static image/data server",

        "cmd": (
            "cd /home/user/jetson-containers/data && "
            f"python3 -m http.server 9000 --bind {t_vllm['ip']}"
        ),
        "stop_cmd": "fuser -k 9000/tcp || true",
    },
    "Display": {
        "target": t_disp,
        "url": f"http://{t_disp['ip']}:{t_disp['port']}",
        "cmd": (
            "cd /home/user/GIT/NanoLLM_VILA_and_OWL && "
            "python3 display_server.py "
            "--root /home/user/jetson-containers/data/R1 "
            f"--host {t_disp['ip']} --port {t_disp['port']} "
            "--latest-only"
        )
    },
}

# --- UI MAIN ---
st.title(f"🚀 {selected_profile.upper()} Infrastructure")
st.info("Connected to Jetson cluster using provided credentials.")

if "selected_log_service" not in st.session_state:
    st.session_state.selected_log_service = None

if "current_log_text" not in st.session_state:
    st.session_state.current_log_text = ""

# --- Summary ---
active_count = 0
service_states = {}

for name, info in SERVICES.items():
    target = info["target"]

    if "container_name" in info:
        res = ssh_exec(
            target,
            f"docker inspect -f '{{{{.State.Running}}}}' {info['container_name']} 2>/dev/null",
            background=False,
        )
        running = res.stdout.strip() == "true"
    else:
        running = check_status(target)

    service_states[name] = running
    if running:
        active_count += 1

m1, m2, m3 = st.columns(3)
m1.metric("Active services", f"{active_count}/{len(SERVICES)}")
m2.metric("Profile", selected_profile)
m3.metric("Primary target", t_vllm["ip"])

st.markdown("### Services")

# --- Service Cards ---
service_items = list(SERVICES.items())
cards_per_row = 3  # change to 4 if you want denser layout

for row_start in range(0, len(service_items), cards_per_row):
    cols = st.columns(cards_per_row)

    for idx, (col, (name, info)) in enumerate(zip(cols, service_items[row_start:row_start + cards_per_row])):
        i = row_start + idx
        target = info["target"]
        running = service_states[name]

        with col:
            with st.container(border=True):
                st.subheader(name)

                status_text = "🟢 Active" if running else "🔴 Stopped"
                st.write(status_text)

                st.caption(f"{target['user']}@{target['ip']}")
                st.caption(f"Port: {target['port']}")

                # Optional short description per service
                if "description" in info:
                    st.caption(info["description"])

                b1, b2, b3 = st.columns(3)

                with b1:
                    if st.button("Launch", key=f"btn_launch_{i}", disabled=running):
                        print(f"DEBUG - Target: {target['user']}@{target['ip']}")
                        print(f"DEBUG - Executing: {info['cmd']}")
                        ssh_exec(target, info["cmd"])
                        st.rerun()

                with b2:
                    if st.button("Stop", key=f"btn_stop_{i}", disabled=not running):
                        stop_cmd = info.get("stop_cmd")

                        if stop_cmd:
                            ssh_exec(target, stop_cmd, background=False, use_sudo=True)
                        elif "container_name" in info:
                            ssh_exec(target, f"docker rm -f {info['container_name']} >/dev/null 2>&1 || true",
                                     background=False)
                        else:
                            ssh_exec(target, f"fuser -k {target['port']}/tcp || true", background=False, use_sudo=True)

                        st.rerun()

                with b3:
                    if st.button("Logs", key=f"btn_logs_{i}"):
                        st.session_state.selected_log_service = name
                        st.session_state.current_log_text = get_log(target, 120)
                        st.rerun()

                if "url" in info:
                    st.link_button("Open", info["url"])
st.markdown("---")
st.markdown("### 🛠 Operations Console")

selected_name = st.session_state.selected_log_service

if selected_name is None:
    st.info("Click Logs on a service card to inspect it here.")
else:
    info = SERVICES[selected_name]
    target = info["target"]

    st.caption(f"Selected: {selected_name} | {target['user']}@{target['ip']}:{target['port']}")

    c1, c2, c3, c4 = st.columns([1, 1, 1, 4])

    with c1:
        log_lines = st.number_input(
            "Lines",
            min_value=20,
            max_value=1000,
            value=120,
            step=20,
            key="selected_log_lines",
        )

    with c2:
        if st.button("Refresh"):
            st.session_state.current_log_text = get_log(target, int(log_lines))

    with c3:
        if st.button("Clear"):
            clear_log(target)
            st.session_state.current_log_text = ""

    with c4:
        st.write("")

    st.code(
        st.session_state.current_log_text[-15000:] if st.session_state.current_log_text else "No log loaded.",
        language="bash",
    )