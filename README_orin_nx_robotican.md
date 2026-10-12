
# **Jetson Orin NX — Robotican Drone Pipeline**

Companion computer running vLLM (Qwen2-VL) + NanoOWL on the Orin NX that ships on the
Robotican drone. Mirrors the AGX pipeline in `README.md`, adapted for this board's
memory constraints and for running fully offline in the field.

**Current IP (bench/dev network):** `172.16.17.8`
**Drone IP (once mounted):** `192.168.130.2`

Every `ssh`/endpoint below uses the current bench IP — once this board is mounted on
the drone, replace `172.16.17.8` with `192.168.130.2` everywhere (nothing else changes;
all services bind to `0.0.0.0` and use `--network host`, so they come up the same way
regardless of the actual address).

```
ssh iai@172.16.17.8
```

---

## 0. One-time board setup (do this before anything else)

### Disable the desktop/GUI session

This board defaults to booting into a graphical session even headless over SSH, which
permanently reserves ~4.5 GiB of the unified memory pool before any container even
starts — enough to make vLLM fail to allocate KV cache. Switch to a text-mode boot:

```bash
sudo systemctl set-default multi-user.target
sudo reboot
```

Verify after reboot, before starting anything:
```bash
tegrastats --interval 1000
```
`RAM` should show well under 1 GiB used at idle (was previously ~4.7 GiB used by the
GUI alone). Ctrl+C to stop.

### jetson-containers (only needed if rebuilding the NanoOWL engine from scratch)

```bash
git clone https://github.com/dusty-nv/jetson-containers
bash jetson-containers/install.sh
```

---

## 1. **VLLM WITH QWEN2-VL:**

Model: `Qwen/Qwen2-VL-2B-Instruct`, served on port `8080`.

**Important — this image does NOT use `~/.cache/huggingface` or `HF_HOME` for its
model cache, despite what the deprecation warning suggests.** It hardcodes its cache
to `/data/models/huggingface`, the same jetson-containers convention NanoOWL uses for
its CLIP cache. You must mount `/data` or every restart re-downloads the full model
(~585s on this network).

Production run (detached, auto-restarts on reboot, fully offline, with a persisted
`torch.compile` cache so startup doesn't recompile every time):

```bash
mkdir -p ~/vllm_compile_cache

docker run -d --restart unless-stopped --name vllm_qwen \
  --runtime nvidia --network host --shm-size=16g \
  -v /home/iai/jetson-containers/data:/data \
  -v ~/vllm_compile_cache:/root/.cache/vllm \
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  -e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
  ghcr.io/nvidia-ai-iot/vllm:latest-jetson-orin \
  vllm serve Qwen/Qwen2-VL-2B-Instruct \
    --port 8080 \
    --gpu-memory-utilization 0.55 \
    --max-model-len 2048 \
    --max-num-seqs 4 \
    --limit-mm-per-prompt '{"image":1,"video":0}' \
    --trust-remote-code
```

These values are tuned so vLLM, NanoOWL and DA3 all fit in the 15.6 GB together. At
`0.62` / 4096 it failed with `ValueError: ... KV cache is needed, which is larger than the
available KV cache memory` once DA3 was loaded (vLLM profiles a video input by default,
reserving memory the pipeline never uses; `video:0` removes that). Start vLLM **first**,
then NanoOWL, then DA3 — after a reboot Docker restarts containers in no fixed order, so
check this on the first boot.

Watch it come up:
```bash
docker logs -f vllm_qwen
```
Look for `Application startup complete.` — on a warm cache this takes well under a
minute; on a cold cache (first run on a freshly flashed board) budget ~15 minutes for
the model download alone.

**If you ever see `ValueError: No available memory for the cache blocks`:** that's a
KV-cache sizing issue, not a download issue — first check the GUI isn't back
(`systemctl get-default` should say `multi-user.target`), then check nothing else is
competing for GPU memory (`docker ps`, `tegrastats`).

**Offline flags explained:** `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1` force
local-cache-only lookups with zero network attempts — required for the drone, since
without them a cache hit still costs a network round-trip for freshness checking,
which would hang on a connection timeout with no network present at all.

Test:
```bash
curl -s http://127.0.0.1:8080/v1/models | python3 -m json.tool

curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2-VL-2B-Instruct",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "Extract ONLY object names from the image.\nOutput a bullet list.\nEach bullet must be exactly: '\''A <object>'\'' or '\''An <object>'\''.\nNo adjectives. No verbs. No extra text."},
        {"type": "image_url", "image_url": {"url": "http://127.0.0.1:9000/R1/latest/some_image.jpg"}}
      ]
    }],
    "max_tokens": 128
  }' | python3 -m json.tool
```

---

## 2. **NanoOWL Object Detector**

Custom image `nanoowl_orin_nx:v1` (built from `dustynv/nanoowl:r36.4.0`, with the
TensorRT image-encoder engine already baked in, plus `flask`/`flask-cors` and
`nanoowl_service.py` already committed into it — see `nanoOWL/nanoowl_service.py` in
this repo for the source).

**Same `/data` mount requirement as vLLM** — this image symlinks `/root/.cache/clip`
to `/data/models/clip` for the CLIP text-encoder checkpoint. Without the mount, this
symlink is dangling and `clip.load()` crashes with `FileExistsError` on startup.

```bash
docker run -d --restart unless-stopped --name now_eng \
  --runtime nvidia --network host --ipc=host \
  -v /home/iai/jetson-containers/data:/data \
  nanoowl_orin_nx:v1 \
  python3 /opt/nanoowl/examples/jetson_server/nanoowl_service.py \
    --engine /opt/nanoowl/data/owl_image_encoder_patch32.engine \
    --host 0.0.0.0 --port 5060 --min-score 0.2
```

```bash
docker logs -f now_eng
```
Look for `NanoOWL ready in ...s` — with the CLIP cache warm this is under 10 seconds.

Already offline-safe by default: the TensorRT engine is fully local (baked into the
image), and OpenAI's CLIP implementation checks the cached file's hash *before*
attempting any download, so a cache hit never touches the network — no extra flags
needed here, unlike vLLM.

Test:
```bash
echo -n '{"image_b64": "' > request.json
base64 -w 0 "/home/iai/R1_20260127_122951.jpg" >> request.json
echo '", "prompts": ["a photo of a chair", "a photo of a bag"], "annotate": false}' >> request.json

curl -s -X POST http://127.0.0.1:5060/infer \
  -H "Content-Type: application/json" \
  --data-binary @request.json | python3 -m json.tool
```

---

## 3. **comm_manager_vllm.py** — end-to-end pipeline

Reads frames from `--captures-root/latest`, asks vLLM for an object list per image,
derives NanoOWL prompts from that list, runs detection, and writes results + an
annotated `_ann.jpg` back into a new timestamped folder under `--captures-root`.

```bash
cd ~/NanoLLM_VILA_and_OWL
python3 comm_manager_vllm.py \
  --no-config \
  --host 0.0.0.0 --port 5050 \
  --captures-root /home/iai/captures/R${DRONE_ID} \
  --endpoint http://127.0.0.1:8080 \
  --nanoowl-endpoint http://127.0.0.1:5060/infer \
  --vllm-url http://127.0.0.1:8080 \
  --vllm-model Qwen/Qwen2-VL-2B-Instruct \
  --vllm-max-tokens 128 \
  --forward-json-url "" \
  --vlm-timeout 60 --retries 3 --retry-sleep 2 \
  --sleep-between 2 --force
```

`--no-config` bypasses `config/networks.yaml`/`--profile` entirely — this device isn't
wired into that profile system (it was built for the `agx1`/`agx2`/`nano` role names
from the AGX-based pipeline), so every endpoint is passed explicitly here, pointed at
`127.0.0.1` since vLLM and NanoOWL both run locally on this same board.

This is currently a **single-shot** run, not a watch loop — rerun the command each
time you want to process a fresh batch of frames.

---

## 4. **Display Server (Web GUI Viewer)**

```bash
cd ~/NanoLLM_VILA_and_OWL
python3 display_server.py --root /home/iai/captures/R${DRONE_ID} --host 0.0.0.0 --port 8090 --latest-only
```

Open in a browser: `http://172.16.17.8:8090` (or `http://192.168.130.2:8090` once on
the drone).

---

## 5. **Depth Anything 3 (metric depth HTTP server)**

`depth_anything_http_server` (from the `daphnaa/ros2-depth-anything-v3-trt` fork, pinned
at `52131cc`) running in a container built from `docker/da3/`. Endpoint:
`POST http://127.0.0.1:5070/bbox_depth` — the same API `comm_manager_vllm.py` calls via
`--depth-endpoint`. Model: `DA3METRIC-LARGE` at 252x448 (fp16, ~90 ms/inference, ~0.7 GB GPU).

**TensorRT engines are device-specific** — never copy an `.engine` from the AGX; copy the
`.onnx` + `.onnx.data` and build the engine on this board. The ONNX looks for its weights
under the *original* export name `DA3METRIC-LARGE.onnx.data`, so keep one folder per
resolution and name the data file exactly that.

Frames are 640x360, so the server runs with `--tiling 0` (one inference per frame) and a
camera YAML for that size (`docker/da3/camera_info_640x360.yaml` = the 4K calibration
divided by 6). **That YAML is only right if the frames come from that same camera** — metric
depth is scaled by the focal length, so use the real camera's calibration otherwise. The
loader reads `image_width`, `image_height` and `camera_matrix.data` (9 values), not `fx`/`fy`.

```bash
# 0. DA3 source is NOT in this repo (docker/da3/src is git-ignored) - fetch it once:
git clone https://github.com/daphnaa/ros2-depth-anything-v3-trt ~/NanoLLM_VILA_and_OWL/docker/da3/src/ros2-depth-anything-v3-trt
git -C ~/NanoLLM_VILA_and_OWL/docker/da3/src/ros2-depth-anything-v3-trt checkout 52131cc

# 1. image (needs internet once)
cd ~/NanoLLM_VILA_and_OWL/docker/da3 && docker build -t da3:orin .

# 2. model folder: ONNX + data (data file renamed!) + camera YAML
mkdir -p ~/da3_models/L-252x448
#   DA3METRIC-LARGE.252x448.onnx       -> ~/da3_models/L-252x448/
#   DA3METRIC-LARGE.252x448.onnx.data  -> ~/da3_models/L-252x448/DA3METRIC-LARGE.onnx.data
cp ~/NanoLLM_VILA_and_OWL/docker/da3/camera_info_640x360.yaml ~/da3_models/L-252x448/cam.yaml

# 3. build the engine on this board
M=~/da3_models/L-252x448; NAME=DA3METRIC-LARGE.252x448
docker run --rm --runtime nvidia -v $M:/models da3:orin \
  /usr/src/tensorrt/bin/trtexec --onnx=/models/$NAME.onnx \
  --saveEngine=/models/$NAME.fp16.engine --fp16

# 4. run
docker run -d --restart unless-stopped --name da3 --runtime nvidia --network host \
  -v ~/da3_models/L-252x448:/models -v /home/iai/captures:/home/iai/captures \
  da3:orin ros2 run depth_anything_v3 depth_anything_http_server \
  --model /models/$NAME.fp16.engine --camera-yaml /models/cam.yaml \
  --max-depth 7.0 --host 0.0.0.0 --port 5070 --save-depth --tiling 0
```

The server writes the depth colormap to `<image_dir>_depth/<name>_depth.jpg`, with
`image_dir` taken from the request and resolved **inside the container** — hence the
`/home/iai/captures` mount at the identical path (files are written as root). Without it the
output stays in the container's own filesystem. Check with
`docker logs da3 | grep -E "SAVE|NOT SAVED"`.

Verified on the board: build, engine, server start, and a 640x360 request (`200`, ~0.14 s).
Not yet verified: the `/home/iai/captures` mount and `comm_manager_vllm.py --depth-endpoint
http://127.0.0.1:5070/bbox_depth` end to end.

Build notes (already handled in `docker/da3/Dockerfile`): the fork tests an undefined
`TENSORRT_VERSION_MAJOR` so it takes the TensorRT<8 path (`buildEngineWithConfig`, removed in
TRT 10) — patched to `NV_TENSORRT_MAJOR`; and `libnvinfer` references DLA libs that only
exist under `--runtime nvidia`, so the build links with `--allow-shlib-undefined`. Do **not**
`sed` the `cv_bridge` includes — the fork ships its own shim for that.

---

## 6. **Startup order (systemd service)** — required for a reliable boot

vLLM, NanoOWL and DA3 share the board's 15.6 GB, and vLLM sizes its GPU budget at startup.
If Docker restarts all three at once after a reboot, NanoOWL/DA3 load while vLLM is still
profiling and vLLM dies with `Free memory ... less than desired GPU memory utilization` /
`No available memory for the cache blocks`, then restart-loops. So only `vllm_qwen` keeps a
Docker restart policy; `orin_nx/start_services.sh` starts the other two **after** vLLM
answers (it also drops the page cache, which CUDA does not count as free on Jetson).

```bash
chmod +x ~/NanoLLM_VILA_and_OWL/orin_nx/start_services.sh
docker update --restart=no now_eng da3          # Docker must not start these early
sudo cp ~/NanoLLM_VILA_and_OWL/orin_nx/orin-nx-services.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable orin-nx-services
```

Check after a reboot (cold boot to all three answering took ~7.5 min, vLLM ~6 of that):
```bash
systemctl status orin-nx-services --no-pager | tail -8
journalctl -u orin-nx-services --no-pager | tail -8
for p in 8080 5060 5070; do printf "$p: "; curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:$p/; done
```
`404` on each port is healthy. The unit's path assumes `/home/iai/NanoLLM_VILA_and_OWL`.
The unrelated `iai` container (idle, ~20 MB) can keep running.

---

## 7. **Drone camera (USB FHD Camera) and capture**

The camera shows up as **two** device nodes: `/dev/video0` is the capture node (MJPG up to
1920x1080 @ 30 fps) and `/dev/video1` is its metadata node (no formats). The
`/dev/HD_CAMERA` udev symlink currently points at `video1` — **wrong**; use `/dev/video0`
until the rule is fixed (it should match `ATTR{index}=="0"`). OpenCV also cannot open a
udev symlink by name, so `auto_capture_interval.py` resolves it with `realpath`.

```bash
v4l2-ctl --list-devices        # find the capture node
cd ~/NanoLLM_VILA_and_OWL
python3 auto_capture_interval.py --root /home/iai/captures/R${DRONE_ID} \
  --device /dev/video0 --width 640 --height 360 --interval 2 --batch-size 20
```
640x360 (the default) is the resolution `docker/da3/camera_info_640x360.yaml` was calibrated
for — capturing at a different size means DA3's metric depth needs a matching recalibration.
Confirmed available at a full 30fps in MJPG via `v4l2-ctl --list-formats-ext` (MJPG is on by
default; pass `--no-mjpg` to disable it).
It prints `[Camera] Streaming at WxH` (what the camera really gave) and writes timestamped
session folders plus the `latest` symlink that `comm_manager_vllm.py` reads.

**The lens is a strong fisheye** (barrel distortion, circular vignette). DA3 and its
focal-length scaling assume a pinhole camera, so depth from these frames is only a rough
relative map, and `cam.yaml` (the 4K camera's calibration / 6) is wrong for it. For real
distances: calibrate this camera with OpenCV's fisheye model and undistort before DA3.

The board has no clock source on the drone network, so the time resets at boot; check that
`latest` still points at the newest run.

---

## 8. **Reaching the board through the RPi**

On the drone network the NX (`192.168.130.2`) is only reachable via the RPi (`192.168.131.103`):
```bash
ssh -J rooster@192.168.131.103 iai@192.168.130.2
scp -J rooster@192.168.131.103 file iai@192.168.130.2:~/NanoLLM_VILA_and_OWL/
rsync -avP -e "ssh -J rooster@192.168.131.103" ./dir/ iai@192.168.130.2:~/dir/
```
The display server uses relative URLs, so a plain port forward works. Start it on the NX
(`--host 0.0.0.0 --port 8090`, see section 4), then on the laptop leave this running and open
`http://localhost:8090` (not the RPi's address):
```bash
ssh -N -L 8090:192.168.130.2:8090 rooster@192.168.131.103
```
`channel N: open failed: Connection refused` in that terminal means the display server is not
running on the NX. VSCode SFTP (`.vscode/sftp.json`) must point at the same address (plus a
`hop` for the RPi) or you upload by hand with `scp -J` — `uploadOnSave` only fires on a VSCode save.

---

## 9. **Identifying which drone you're on**

R1/R2/R3's Orin NX boards are identical clones — same IP (`192.168.130.2`), same username
(`iai`), same hostname (`ubuntu`) — so a bare `iai@ubuntu:~$` prompt gives no clue which
board an SSH session is actually on. Rather than inventing a separate label, reuse the
`ROS_DOMAIN_ID` the existing `iai` ROS2 container already reads from `~/iai_ws/src/docker/.env`
at build time — in this fleet's convention it already equals the drone number (the RPi's own
prompt shows this as `[R3@3]`: drone R3, domain 3).

```bash
sudo tee /etc/update-motd.d/99-drone-id > /dev/null <<'EOF'
#!/bin/bash
# /etc/update-motd.d/ scripts run via PAM as root, not as the login user, so $HOME
# here is /root, not /home/iai - hardcode the path rather than relying on $HOME.
ENV_FILE="/home/iai/iai_ws/src/docker/.env"
if [ -f "$ENV_FILE" ]; then
  DOMAIN=$(grep '^ROS_DOMAIN_ID=' "$ENV_FILE" | cut -d= -f2)
fi
echo
echo " ##########################"
echo " ####   THIS IS  R${DOMAIN:-?}    ####"
echo " ##########################"
echo
EOF
sudo chmod +x /etc/update-motd.d/99-drone-id

cat >> ~/.bashrc <<'EOF'
export DRONE_ID=$(grep '^ROS_DOMAIN_ID=' "$HOME/iai_ws/src/docker/.env" 2>/dev/null | cut -d= -f2)
PS1="[R${DRONE_ID:-?}] ${PS1}"
EOF
source ~/.bashrc   # applies immediately; the MOTD banner shows on the next fresh login
```

The `.bashrc` snippet keeps `$HOME` since it genuinely runs as the login user there. Both
read `.env` dynamically rather than hardcoding a board name, so this is the same two
commands on every board — it just reflects whatever that board's own `.env` already says.
Verified on R2: prompt becomes `[R2] iai@ubuntu:~$` (the MOTD needed the `$HOME` fix above
after it first showed `R?` — PAM's root context was the culprit).

**`$DRONE_ID` is also the single source of truth for this board's captures path** — every
`--captures-root`/`--root` below uses `/home/iai/captures/R${DRONE_ID}`, so a board's frames
always live under its own real drone number rather than a leftover generic `R1` folder name.

---

## Deploy workflow (editing this repo)

This repo is synced to the Orin NX via the VSCode **SFTP** extension
(`.vscode/sftp.json`), not git — editing locally and saving auto-uploads to
`/home/iai/NanoLLM_VILA_and_OWL/` on the board. Update `sftp.json`'s `host` field when
the IP changes to `192.168.130.2`.

An edit made by a tool/script directly (not a VSCode `Ctrl+S`) does **not** trigger
`uploadOnSave` — push it manually:
```bash
rsync -avz /path/to/changed_file.py iai@172.16.17.8:/home/iai/NanoLLM_VILA_and_OWL/
```

---

## Backups and cloning to more boards

Everything the three services need lives outside git: three Docker images (vLLM, NanoOWL,
DA3 — `nanoowl_orin_nx:v1` and `da3:orin` exist **only** as local images) and the model
caches under `~/jetson-containers/data` and `~/da3_models`. None of it survives a reflash.
The same export is how you set up another board when you can only reach one at a time
(via a laptop or USB disk). Budget ~35 GB.

**On the source board** — write to `~/export`, **not** the repo folder (gzip is slow here;
each step takes minutes, wait for the prompt to return before the next):
```bash
mkdir -p ~/export && cd ~/export
docker save ghcr.io/nvidia-ai-iot/vllm:latest-jetson-orin | gzip > vllm.tar.gz
docker save nanoowl_orin_nx:v1 | gzip > nanoowl.tar.gz
docker save da3:orin | gzip > da3.tar.gz
tar czf data.tar.gz -C ~ jetson-containers/data da3_models
sha256sum *.tar.gz > SHA256SUMS
```
(`vllm_compile_cache` is only a startup-time cache and is root-owned, so it is left out;
vLLM rebuilds it on first start.)

**To the laptop / disk, then to the new board** (`rsync` resumes if interrupted):
```bash
rsync -avP iai@<source-ip>:~/export/ ~/jetson_export/
rsync -avP ~/jetson_export/ iai@<new-ip>:~/export/
```

**On the new board** (needs JetPack, Docker with the nvidia runtime, and section 0 done):
```bash
cd ~/export && sha256sum -c SHA256SUMS
for f in vllm nanoowl da3; do gunzip -c $f.tar.gz | docker load; done
tar xzf data.tar.gz -C ~
```
Then run sections 1, 2 and 5 in that order (vLLM first — see section 1). Per board:
delete the copied `~/da3_models/*/*.engine` and rebuild it with the `trtexec` step in
section 5 (engines are tied to the exact GPU/TensorRT), set the new IP, put that
camera's calibration in `cam.yaml`, and set up the drone-ID banner/prompt (section 9)
so an SSH session always shows which board you're on.

---

## Known issues / not yet set up on this board

- NanoOWL prompt quality depends on caption phrasing from the VLM; bare nouns
  ("chair") detect worse than templated phrases ("a photo of a chair"). The VLM's
  bullet-list prompt in `comm_manager_vllm.py`'s `call_vlm()` already asks for simple
  `"A <object>"` phrasing, and `caption_to_owl_prompts()` splits on both newlines and
  commas to handle however the model happens to format its answer.
