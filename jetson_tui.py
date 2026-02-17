import os
import subprocess
import sys
import time
import signal

# --- CONFIGURATION ---
JETSON_HOST = "192.168.131.22"
JETSON_USER = "user"
JETSON_PASS = "1"
REMOTE_PATH = "~/GIT/NanoLLM_VILA_and_OWL"
CAPTURES_ROOT = "/home/user/jetson-containers/data/R1/"


def clear():
    os.system('cls' if os.name == 'nt' else 'clear')


def run_comm_manager_auto(cmd_string):
    print(f"\n--- Launching Comm Manager (Run-Once) ---")
    full_cmd = f"cd {REMOTE_PATH} && {cmd_string}"
    ssh_call = [
        "sshpass", "-p", JETSON_PASS,
        "ssh", "-o", "StrictHostKeyChecking=no",
        f"{JETSON_USER}@{JETSON_HOST}",
        full_cmd
    ]

    process = subprocess.Popen(
        ssh_call,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

    summary_line = None

    try:
        for line in process.stdout:
            print(line, end='')

            # Detect completion summary from comm_manager
            if "[worker] summary:" in line:
                summary_line = line.strip()
                # Nice, short UI message for exhibition
                print("\n✅ Processing finished:", summary_line)

    except KeyboardInterrupt:
        # If you stop the TUI, try to stop the remote process too
        process.send_signal(signal.SIGINT)

    rc = process.wait()

    if summary_line is None:
        print("\n⚠️ Comm Manager exited without summary line.")
    else:
        print("\n[TUI] Ready for another run.")

    return rc, summary_line



def cleanup_remote_port(port):
    """Force kills any process on a specific port on the Jetson."""
    print(f"[*] Cleaning up Port {port}...")
    cmd = f"echo '{JETSON_PASS}' | sudo -S fuser -k {port}/tcp || true"
    ssh_call = ["sshpass", "-p", JETSON_PASS, "ssh", "-o", "StrictHostKeyChecking=no", "-t", f"{JETSON_USER}@{JETSON_HOST}", cmd]
    # Use call to wait for it to finish
    subprocess.call(ssh_call, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def run_ssh(cmd_string, title):
    print(f"\n--- Launching {title} ---")
    full_cmd = f"cd {REMOTE_PATH} && {cmd_string}"
    ssh_call = ["sshpass", "-p", JETSON_PASS, "ssh", "-o", "StrictHostKeyChecking=no",
                f"{JETSON_USER}@{JETSON_HOST}", full_cmd]
    return subprocess.call(ssh_call)


def main():
    cap_cmd = f"python3 capture_on_enter.py --root {CAPTURES_ROOT} --rows 1 --cols 1 --capture-now"

    comm_cmd = (
        f"python3 comm_manager_vllm.py --profile robotican "
        f"--vllm-model espressor/meta-llama.Llama-3.2-3B-Instruct_W4A16 "
        f"--captures-root {CAPTURES_ROOT} --endpoint http://192.168.131.22:8080 "
        f"--host 192.168.131.22 --force --depth-endpoint http://192.168.131.22:5070/bbox_depth "
        f"--run-once"
    )

    while True:
        clear()
        print("==========================================================")
        print(f"   JETSON ONE-CLICK PIPELINE | {JETSON_USER}@{JETSON_HOST}")
        print("==========================================================")
        print("Press [ENTER] to CAPTURE + PROCESS")
        print("Type  q  then ENTER to quit")
        print("----------------------------------------------------------")

        choice = input("> ").strip().lower()
        if choice == "q":
            sys.exit(0)

        # Stage 1: Capture
        rc = run_ssh(cap_cmd, "Stage 1: Capture (one-shot)")
        if rc != 0:
            print("\n[TUI] Capture failed. Press Enter to retry...")
            input()
            continue

        # Stage 2: Process
        run_comm_manager_auto(comm_cmd)

        print("\n✅ Ready for another run.")
        time.sleep(2)



if __name__ == "__main__":
    main()