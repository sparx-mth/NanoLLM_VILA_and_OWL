
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
    --gpu-memory-utilization 0.62 \
    --max-model-len 4096 \
    --trust-remote-code
```

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
  --captures-root /home/iai/captures/R1 \
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
python3 display_server.py --root /home/iai/captures/R1 --host 0.0.0.0 --port 8090 --latest-only
```

Open in a browser: `http://172.16.17.8:8090` (or `http://192.168.130.2:8090` once on
the drone).

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

## Backups (do this before the board goes on the drone)

Both the vLLM model cache and the NanoOWL CLIP cache live under
`/home/iai/jetson-containers/data/models/`, outside any container's own filesystem —
survives container restarts, but **not** a board reflash. Back them up externally:

```bash
# from your dev machine
scp -r iai@172.16.17.8:/home/iai/jetson-containers/data/models/huggingface /path/to/backups/
scp -r iai@172.16.17.8:/home/iai/jetson-containers/data/models/clip /path/to/backups/

# the NanoOWL image itself only exists as a local docker image — save it too
ssh iai@172.16.17.8 "docker save nanoowl_orin_nx:v1 | gzip" > /path/to/backups/nanoowl_orin_nx_v1.tar.gz
```

To restore onto a freshly flashed board (no internet needed): `scp` these back into
place at the same paths, `docker load < nanoowl_orin_nx_v1.tar.gz`, then run the
commands in sections 1–2 above as normal — they'll find the caches already warm.

---

## Known issues / not yet set up on this board

- **Depth Anything v3** (`--depth-endpoint`) is not installed here — `comm_manager_vllm.py`
  logs a harmless connection-refused error for it and continues without depth data.
- NanoOWL prompt quality depends on caption phrasing from the VLM; bare nouns
  ("chair") detect worse than templated phrases ("a photo of a chair"). The VLM's
  bullet-list prompt in `comm_manager_vllm.py`'s `call_vlm()` already asks for simple
  `"A <object>"` phrasing, and `caption_to_owl_prompts()` splits on both newlines and
  commas to handle however the model happens to format its answer.
