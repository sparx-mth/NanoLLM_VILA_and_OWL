# **System Overview – VILA + Jetson LLM + NanoOWL Integrated Pipeline**

```
📷 Camera / Stream (/dev/video0)
       │
       ▼
🧩 capture_frames.py → /opt/missions/poses.json
       │
       │ Sends image_path requests to
       ▼
🧠 VILA API Server (main_with_time_and_json_and_image_http.py)
(run from: /opt/NanoLLM/nano_llm/chat/__main__.py)
       │
       │ Generates textual description for each image
       ▼
🌈 display_server.py (Web GUI Viewer)
Displays live image grid from /home/user/jetson-containers/data/images/captures/
       └── http://<DEVICE_IP>:8090  ← Live dashboard for images + captions
       │
       │ Sends captions to
       ▼
🖥️ comm_manager.py 
       │
       │ Forwards description to LLM on Jetson #2 → extracts object list
       ▼
🤖 NanoOWL (Object Detection Engine)
       │
       │ Receives the image + object list → returns bounding boxes
       ▼
🎨 Automatic OpenCV Annotator
       │
       └── Saves <basename>_ann.jpg next to each original image
             (with BBOX and labels)
```

This creates a **real-time, closed-loop multimodal system** connecting:
**camera capture → vision-language description → object extraction → bounding-box detection → annotated visualization**.





## 🔹 **Pipeline Stages**
first- get in to the jetson:

```
ssh -X user@172.16.17.12
```
### 1. **VILA API Server**
**Run inside the VILA container:**
```bash
jetson-containers run -it   --publish 8080:8080   --volume /home/user/jetson-containers/data:/mnt/VLM/jetson-data   nano_llm_custom /bin/bash
```

Then start the API server:
```bash
python3 -m nano_llm.chat   --api=mlc   --model Efficient-Large-Model/VILA1.5-3b   --max-context-len 256   --max-new-tokens 32   --save-json-by-image   --server --port 8080 --notify-url http://172.16.17.12:5050/from_vila
```
### 2. **NanoOWL Object Detector**

```bash
sudo docker run -it --network host nanoowl_new:v1.4 /bin/bash
```

 ```bash
cd examples/jetson_server/
python3 nanoowl_service.py \
  --engine /opt/nanoowl/data/owl_image_encoder_patch32.engine \
  --host 0.0.0.0 --port 5060
```

### 3. **Display Server (Web GUI Viewer)**

**Run:**
```bash
cd shir
python3 display_server_2.py   --root /home/user/jetson-containers/data/images/captures   --host 0.0.0.0   --port 8090   --latest-only
```
** if you run from local computer:
```bash
cd /home/user1/shir/22-10-25/
python3 display_server.py \
  --root $HOME/mnt/jetson_captures \
  --host 0.0.0.0 \
  --port 8090 \
  --scan-interval 2.0 \
  --latest-only
```

### 4. **comm_manager.py**
**Run:**
```bash
cd shir
python3 comm_manager_2.py   --host 0.0.0.0 --port 5050   --jetson2-endpoint http://172.16.17.11:5050/prompts   --captures-root /home/user/jetson-containers/data/images/captures   --nanoowl-endpoint http://172.16.17.12:5060/infer   --forward-timeout 25   --forward-retries 7   --nanoowl-timeout 70   --nanoowl-annotate 0 --forward-json-url http://172.16.17.9:9090/ingest 

```


### 5. `capture_frames.py`
**How to Run:**
```bash
cd /home/user/jetson-containers/data/images
 python3 capture_frames.py   --source /dev/video0   --vlm http://172.16.17.12:8080/describe --interactive --crop-frac 0.75 --sleep 15
```
```bash
 python3 capture_frames.py --source /dev/video1 --frames-dir /home/user/jetson-containers/data/images/captures/2025_10_19___17_18_28/ --loop-sleep 65 --vlm http://172.16.17.12:8080/describe 
```
