#!/usr/bin/env python3
import argparse
import os
import sys
import yaml

def die(msg: str):
    print(f"[config] {msg}", file=sys.stderr)
    sys.exit(1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/networks.yaml")
    ap.add_argument("--profile", default=None, help="adsl | robotican")
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    profiles = cfg.get("profiles", {})
    default_profile = cfg.get("defaults", {}).get("profile")
    profile = args.profile or os.environ.get("R2_PROFILE") or default_profile
    if profile not in profiles:
        die(f"Unknown profile '{profile}'. Options: {list(profiles.keys())}")

    p = profiles[profile]
    hosts = p["hosts"]
    services = p["services"]

    def url(svc_name: str, path: str = ""):
        svc = services[svc_name]
        host_key = svc["host"]
        ip = hosts[host_key]["ip"]
        port = svc["port"]
        return f"http://{ip}:{port}{path}"

    env = {
        "R2_PROFILE": profile,

        # Hosts
        "AGX1_IP": hosts["agx1"]["ip"],
        "AGX1_USER": hosts["agx1"]["user"],
        "AGX2_IP": hosts["agx2"]["ip"],
        "AGX2_USER": hosts["agx2"]["user"],
        "NANO_IP": hosts["nano"]["ip"],
        "NANO_USER": hosts["nano"]["user"],

        # Services (URLs)
        "COMM_MANAGER_URL": url("comm_manager"),
        "VILA_URL": url("vila_api"),
        "VILA_DESCRIBE_URL": url("vila_api", "/describe"),
        "NANOOWL_URL": url("nanoowl", "/infer"),
        "VLLM_URL": url("vllm"),
        "DISPLAY_URL": url("display_server"),
        "INGEST_JSON_URL": url("ingest_json", "/ingest"),
    }

    # Print as bash exports (so you can: eval "$(python3 tools/select_profile.py --profile adsl)")
    for k, v in env.items():
        print(f'export {k}="{v}"')

if __name__ == "__main__":
    main()
