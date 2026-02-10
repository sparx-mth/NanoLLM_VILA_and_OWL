from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
import os
import yaml


@dataclass(frozen=True)
class NetProfile:
    name: str
    agx1_ip: str
    agx1_user: str
    agx2_ip: str
    agx2_user: str
    nano_ip: str
    nano_user: str

    comm_manager_url: str
    vila_url: str
    vila_describe_url: str
    nanoowl_infer_url: str
    vllm_url: str
    display_url: str
    ingest_json_url: str


def _url(host_ip: str, port: int, path: str = "") -> str:
    return f"http://{host_ip}:{port}{path}"


def load_profile(
    config_path: str | Path = "config/networks.yaml",
    profile_name: Optional[str] = None,
) -> NetProfile:
    config_path = Path(config_path)

    with config_path.open("r") as f:
        cfg: Dict[str, Any] = yaml.safe_load(f)

    profiles = cfg.get("profiles", {})
    default_profile = cfg.get("defaults", {}).get("profile")

    name = (
        profile_name
        or os.environ.get("R2_PROFILE")
        or default_profile
    )
    if not name or name not in profiles:
        raise ValueError(f"Unknown profile '{name}'. Options: {list(profiles.keys())}")

    p = profiles[name]
    hosts = p["hosts"]
    services = p["services"]

    def host_ip(host_key: str) -> str:
        return hosts[host_key]["ip"]

    def svc_url(svc_name: str, path: str = "") -> str:
        svc = services[svc_name]
        ip = host_ip(svc["host"])
        return _url(ip, int(svc["port"]), path)

    return NetProfile(
        name=name,

        agx1_ip=hosts["agx1"]["ip"],
        agx1_user=hosts["agx1"]["user"],
        agx2_ip=hosts["agx2"]["ip"],
        agx2_user=hosts["agx2"]["user"],
        nano_ip=hosts["nano"]["ip"],
        nano_user=hosts["nano"]["user"],

        comm_manager_url=svc_url("comm_manager"),
        vila_url=svc_url("vila_api"),
        vila_describe_url=svc_url("vila_api"),
        #vila_describe_url=svc_url("vila_api", "/describe"),
        nanoowl_infer_url=svc_url("nanoowl", "/infer"),
        vllm_url=svc_url("vllm"),
        display_url=svc_url("display_server"),
        ingest_json_url=svc_url("ingest_json", "/ingest"),
    )
