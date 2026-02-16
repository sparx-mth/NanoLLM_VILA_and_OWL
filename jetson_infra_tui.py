import os
import subprocess
import time

# --- CONFIGURATION ---
JETSON_HOST = "192.168.131.22"
JETSON_USER = "user"
JETSON_PASS = "1"
DATA_PATH = "/home/user/jetson-containers/data"
REPO_PATH = "/home/user1/GIT/NanoLLM_VILA_and_OWL"


def run_ssh_tmux(session_name, cmd):
    print(f"[*] Starting {session_name}...")
    # Prepend 'export TERM=xterm-256color' to the command
    full_cmd = f"export TERM=xterm-256color && tmux new-session -d -s {session_name} \"{cmd}\""
    ssh_call = ["sshpass", "-p", JETSON_PASS, "ssh", "-o", "StrictHostKeyChecking=no", f"{JETSON_USER}@{JETSON_HOST}", full_cmd]
    subprocess.run(ssh_call)


def main():
    while True:
        os.system('clear')
        print("==========================================================")
        print("   JETSON INFRASTRUCTURE: DOCKER & SERVICE MANAGER")
        print("==========================================================")
        print("1) START vLLM (Brain - Port 8080)")
        print("2) START NanoOWL (Eyes - Port 5060)")
        print("3) START Display Servers (Gallery/9000 + Latest/8090)")
        print("4) START ALL INFRASTRUCTURE")
        print("----------------------------------------------------------")
        print("s) STOP ALL (Kill all Docker & Tmux)")
        print("j) jtop (Monitor RAM/GPU)")
        print("q) Exit")
        print("----------------------------------------------------------")

        choice = input("Select: ").strip().lower()

        # vLLM Command (with the serving command inside)
        vllm_cmd = (
            "docker run --rm --runtime nvidia --network host "
            "-v ~/.cache/huggingface:/root/.cache/huggingface "
            "vllm_qwen3_vl_4b_instruct_aws_4bit:latest vllm serve cpatonn/Qwen3-VL-4B-Instruct-AWQ-4bit "
            "--host 0.0.0.0 --port 8080 --dtype float16 --gpu-memory-utilization 0.5 "
            "--max-model-len 2048 --max-num-batched-tokens 128 --max-num-seqs 4 --swap-space 0 --enforce-eager"
        )

        # NanoOWL Command
        nano_cmd = (
            "docker run --rm --name now_eng --runtime nvidia --network host --ipc=host "
            "-e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=all "
            "nanoowl_new:v1.5 /bin/bash -c "
            "'cd examples/jetson_server && python3 nanoowl_service.py "
            "--engine /opt/nanoowl/data/owl_image_encoder_patch32.engine --host 0.0.0.0 --port 5060 --min-score 0.2'"
        )

        if choice == '1':
            run_ssh_tmux("vllm", vllm_cmd)
            time.sleep(2)
        elif choice == '2':
            run_ssh_tmux("nanoowl", nano_cmd)
            time.sleep(2)
        elif choice == '3':
            # Kill port 8090 and 9000 before starting to ensure they bind correctly
            cleanup_cmd = f"echo '{JETSON_PASS}' | sudo -S fuser -k 8090/tcp 9000/tcp || true"
            subprocess.run(["sshpass", "-p", JETSON_PASS, "ssh", f"{JETSON_USER}@{JETSON_HOST}", cleanup_cmd])

            run_ssh_tmux("gallery", f"cd {DATA_PATH} && python3 -m http.server 9000 --bind 0.0.0.0")
            run_ssh_tmux("display8090",
                         f"cd {REPO_PATH} && python3 display_server.py --root {DATA_PATH}/R1 --host 0.0.0.0 --port 8090 --latest-only")
            print("[*] Display Servers Launched (Ports 9000 & 8090)")
            time.sleep(1)
            time.sleep(1)
        elif choice == '4':
            run_ssh_tmux("vllm", vllm_cmd)
            run_ssh_tmux("nanoowl", nano_cmd)
            run_ssh_tmux("gallery", f"cd {DATA_PATH} && python3 -m http.server 9000 --bind 0.0.0.0")
            run_ssh_tmux("display8090",
                         f"cd {REPO_PATH} && python3 display_server.py --root {DATA_PATH}/R1 --host 0.0.0.0 --port 8090 --latest-only")
            time.sleep(2)
        elif choice == 's':
            print("[*] Performing Hard Reset of all services...")
            # We add -tt to force a terminal and use echo to feed the password to sudo -S
            stop_cmd = (
                f"tmux kill-server; "
                f"docker stop now_eng vllm_brain 2>/dev/null || true; "
                f"docker rm now_eng vllm_brain 2>/dev/null || true; "
                f"echo '{JETSON_PASS}' | sudo -S fuser -k 5050/tcp 5060/tcp 5070/tcp 8080/tcp 8090/tcp 9000/tcp || true"
            )
            # Added '-tt' to the ssh arguments here
            ssh_call = ["sshpass", "-p", JETSON_PASS, "ssh", "-tt", f"{JETSON_USER}@{JETSON_HOST}", stop_cmd]
            subprocess.run(ssh_call)
            print("\n[!] All Infrastructure Force-Stopped and Ports Cleared.")
            time.sleep(2)
        elif choice == 'j':
            os.system(f"sshpass -p {JETSON_PASS} ssh -t {JETSON_USER}@{JETSON_HOST} 'jtop'")
        elif choice == 'q':
            break


if __name__ == "__main__":
    main()