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


def run_ssh_interactively(cmd_string, title):
    """Used for Capture: Needs full keyboard passthrough."""
    print(f"\n--- Launching {title} ---")
    full_cmd = f"export TERM=xterm-256color && cd {REMOTE_PATH} && {cmd_string}"
    ssh_call = ["sshpass", "-p", JETSON_PASS, "ssh", "-t", "-t", f"{JETSON_USER}@{JETSON_HOST}", full_cmd]
    subprocess.call(ssh_call)


def run_comm_manager_auto(cmd_string):
    """
    Used for Comm Manager: Monitors output and auto-exits
    when it sees the processing is finished.
    """
    print(f"\n--- Launching Comm Manager (Auto-Exit Enabled) ---")
    full_cmd = f"cd {REMOTE_PATH} && {cmd_string}"

    # We use Popen to read the output stream line-by-line
    ssh_call = ["sshpass", "-p", JETSON_PASS, "ssh", f"{JETSON_USER}@{JETSON_HOST}", full_cmd]

    process = subprocess.Popen(ssh_call, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    try:
        for line in process.stdout:
            print(line, end='')  # Print Jetson output to our screen

            # The Magic Trigger:
            if "summary: done=4" in line:
                print("\n[TUI] Detected processing finish! Sending Shutdown...")
                # Send Ctrl+C to the remote process
                process.send_signal(signal.SIGINT)
                break
    except KeyboardInterrupt:
        process.send_signal(signal.SIGINT)

    process.wait()
    print("\n[TUI] Comm Manager Closed. Returning to menu...")

def cleanup_remote_port(port):
    """Force kills any process on a specific port on the Jetson."""
    print(f"[*] Cleaning up Port {port}...")
    cmd = f"echo '{JETSON_PASS}' | sudo -S fuser -k {port}/tcp || true"
    ssh_call = ["sshpass", "-p", JETSON_PASS, "ssh", "-o", "StrictHostKeyChecking=no", "-t", f"{JETSON_USER}@{JETSON_HOST}", cmd]
    # Use call to wait for it to finish
    subprocess.call(ssh_call, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def main():
    while True:
        clear()
        print("==========================================================")
        print(f"   JETSON REMOTE MANAGER | {JETSON_USER}@{JETSON_HOST}")
        print("==========================================================")
        print("1) [CAPTURE] Take 4K Images (Manual exit with 'q')")
        print("2) [COMM]    Run Comm Manager (Auto-exits when done)")
        print("3) [AUTO]    Full Pipeline (Zero-Touch)")
        print("q) Quit")
        print("----------------------------------------------------------")

        choice = input("Select an option: ").strip().lower()

        comm_cmd = (
            f"python3 comm_manager_vllm.py --profile robotican "
            f"--vllm-model espressor/meta-llama.Llama-3.2-3B-Instruct_W4A16 "
            f"--captures-root {CAPTURES_ROOT} --endpoint http://192.168.131.22:8080 "
            f"--host 192.168.131.22 --force --depth-endpoint http://192.168.131.22:5070/bbox_depth"
        )

        if choice == '1':
            cap_cmd = f"python3 capture_on_enter.py --root {CAPTURES_ROOT} --rows 1 --cols 1"
            run_ssh_interactively(cap_cmd, "Image Capture")
            input("\nPress Enter to return to menu...")

        elif choice == '2':
            cleanup_remote_port(5050)  # Kill old Comm Manager
            run_comm_manager_auto(comm_cmd)
            time.sleep(2)

        elif choice == '3':
            # Stage 1: Capture
            cap_cmd = f"python3 capture_on_enter.py --root {CAPTURES_ROOT} --rows 1 --cols 1"
            run_ssh_interactively(cap_cmd, "Stage 1: Capture")

            # Stage 2: Process (Now with Auto-Exit)
            print("\nMoving to Stage 2: Automatic Inference...")
            cleanup_remote_port(5050)  # Kill old Comm Manager
            run_comm_manager_auto(comm_cmd)
            input("\nPipeline Finished. Press Enter...")

        elif choice == 'q':
            sys.exit()


if __name__ == "__main__":
    main()