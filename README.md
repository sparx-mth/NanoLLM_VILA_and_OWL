
# **Pipeline Stages**

## 1. **VILA API Server**
first- get in to the jetson:
```
ssh -X user@172.16.17.12
```
**Run inside the VILA container:**
```bash
jetson-containers run -it   --publish 8080:8080   --volume TODO:edit volume  nano_llm_custom /bin/bash

```

Then start the API server:
```bash
python3 -m nano_llm.chat   --api=mlc   --model Efficient-Large-Model/VILA1.5-3b   --max-context-len 256   --max-new-tokens 32   --save-json-by-image   --server --port 8080 --notify-url http://172.16.17.12:5050/from_vila
```
test:
```bash 
curl -s -X POST http://127.0.0.1:8080/describe   -H "Content-Type: application/json"   -d '{"image_path":"/mnt/VLM/jetson-data/PortraitA_01.jpg"}'
```
## 2. **NanoOWL Object Detector**
first- get in to the jetson:
```
ssh -X user@172.16.17.12
```

```bash
docker run -it --name now_eng \
  --runtime nvidia \
  --network host --ipc=host \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e LD_LIBRARY_PATH=/usr/local/lib:/usr/lib/aarch64-linux-gnu:/usr/lib:/lib \
  nanoowl_new:v1.5 /bin/bash
```

 ```bash
cd examples/jetson_server/
python3 nanoowl_service.py \
  --engine /opt/nanoowl/data/owl_image_encoder_patch32.engine \
  --host 0.0.0.0 --port 5060 --min-score 0.2
```
test
```bash
curl -s -X POST http://172.16.17.12:5060/infer   
-F 'image=@/home/user/Pictures/PortraitA_01.jpg'   
-F 'prompts=["sky","a tree","a bulk"]'  
-F 'annotate=1' | python3 -c 'import sys,json; print(json.dumps(json.load(sys.stdin), indent=2))'


```
## 3. **Display Server (Web GUI Viewer)**

first- get in to the jetson:

```
ssh -X user@172.16.17.12
```
**Run:**
```bash
cd ~/GIT/NanoLLM_VILA_and_OWL
python3 display_server_2.py \
  --root /tmp/incoming_frames/ \
  --host 0.0.0.0 \
  --port 8090 \
  --latest-only
```


## 4. **comm_manager.py**

first- get in to the jetson:

```
ssh -X user@172.16.17.12
```
**Run:**
```bash
cd ~/GIT/NanoLLM_VILA_and_OWL
python3 comm_manager_2.py \
  --host 0.0.0.0 \
  --port 5050 \
  --jetson2-endpoint http://172.16.17.10:5050/prompts \
  --captures-root /tmp/incoming_frames/ \
  --nanoowl-endpoint http://172.16.17.12:5060/infer \
  --forward-timeout 25 \
  --forward-retries 7 \
  --nanoowl-timeout 70 \
  --nanoowl-annotate 0 \
  --forward-json-url http://172.16.17.15:9090/ingest
 ```

## 5. **capture_frames.py**
first- get in to the jetson:

```
ssh -X user@172.16.17.12
```
**How to Run:**
```bash
cd ~/GIT/raspi/
 python3 receiver_vlm.py --host 0.0.0.0 --port 5001
```

call for capture from raspberry Pi 
```bash
cd ~/GIT/raspi/
./discover_and_capture.py --name test --count 8 --interval 15
```
from folder:
```bash
 python3 capture_frames_folder.py --source /dev/video0 --frames-dir /tmp/incoming_frames/2025_10_21___15_37_21/ --loop-sleep 15 --vlm http://172.16.17.12:8080/describe

```
live from auto move: NOT RELEVANT FROM RASPI
```bash
python3 capture_frames.py   --source /dev/video0   --poses /opt/missions/poses.json   --gpio-pin 18 --gpio-edge rising --gpio-pull up --gpio-debounce-ms 50   --out captures --crop-frac 0.7   --vlm http://172.16.17.12:8080/describe --flip-180 --gpio-pin 12 --gpio-first-frame 16
```

## 6. **LLM Object List Extractor**

Connect to Jetson #2:
```bash
ssh user@172.16.17.10
```
in terminal 2:
```bash
cd GIT/NanoLLM_VILA_and_OWL
gunicorn -w 1 -k gthread --threads 8 --timeout 120 -b 0.0.0.0:5050 prompt_converter_llm_v2:app
```
test 
```bash
curl -s http://172.16.17.10:5050/prompts \
  -H "Content-Type: application/json" \
  -d '{"caption":"two black suitcases with red and white labels on the ground"}'
  ```

## 7. **Room Mapping + LLM Navigation Interface (Jetson #3 – 172.16.17.15)**
Connect to Jetson #3:
```bash
ssh nvidia@172.16.17.15
```
Terminal 1 – Start Ollama Server
```bash
ollama serve
```

 * if ollama not install - run : 
```bash
ollama run llama3.1:8b
ollama run llama3.2:3b 
```

Terminal 2 – Launch Room Mapping
```bash
cd ~/GIT/TheAgency/src/room_mapping
source .venv/bin/activate
pip3 install requirements.txt
python3 run_llm_with_web.py
```

Then open in your browser:
```bash
http://172.16.17.15:8080/
```

