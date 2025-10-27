
# **Pipeline Stages**

## 1. **VILA API Server**
first- get in to the jetson:
```
ssh -X user@172.16.17.12
```
**Run inside the VILA container:**
```bash
jetson-containers run -it \
  --publish 8080:8080 \
  --volume /mnt/VLM/jetson-data:/home/user/jetson-containers/data \
  --volume /mnt/VLM:/mnt/VLM \
  nano_llm_custom /bin/bash
```

Then start the API server:
```bash
python3 -m nano_llm.chat   --api=mlc   --model Efficient-Large-Model/VILA1.5-3b   --max-context-len 256   --max-new-tokens 32   --save-json-by-image   --server --port 8080 --notify-url http://172.16.17.12:5050/from_vila
```
## 2. **NanoOWL Object Detector**
first- get in to the jetson:
```
ssh -X user@172.16.17.12
```

```bash
sudo docker run -it --network host nanoowl_new:v1.4 /bin/bash
```

 ```bash
cd examples/jetson_server/
python3 nanoowl_service.py \
  --engine /opt/nanoowl/data/owl_image_encoder_patch32.engine \
  --host 0.0.0.0 --port 5060 --min-score 0.2
```

## 3. **Display Server (Web GUI Viewer)**

first- get in to the jetson:

```
ssh -X user@172.16.17.12
```
**Run:**
```bash
cd ~/shir
python3 display_server_2.py \
  --root /mnt/VLM/jetson-data/images/captures \
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
cd ~/shir
python3 comm_manager_2.py \
  --host 0.0.0.0 \
  --port 5050 \
  --jetson2-endpoint http://172.16.17.11:5050/prompts \
  --captures-root /mnt/VLM/jetson-data/images/captures \
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
cd /mnt/VLM/jetson-data/images
 python3 capture_frames.py   --source /dev/video0   --vlm http://172.16.17.12:8080/describe --interactive --crop-frac 0.75 --sleep 15
```

from folder:
```bash
 python3 capture_frames_folder.py --source /dev/video0 --frames-dir /home/user/jetson-containers/data/images/captures/2025_10_21___15_37_21/ --loop-sleep 15 --vlm http://172.16.17.12:8080/describe

```
live from auto move:
```bash
python3 capture_frames.py   --source /dev/video0   --poses /opt/missions/poses.json   --gpio-pin 18 --gpio-edge rising --gpio-pull up --gpio-debounce-ms 50   --out captures --crop-frac 0.7   --vlm http://172.16.17.12:8080/describe --flip-180 --gpio-pin 12 --gpio-first-frame 16
```

## 6. **LLM Object List Extractor**

Connect to Jetson #2:
```bash
ssh user@172.16.17.11
```
in terminal 1:
```bash
ollama serve
```

in terminal 2:
```bash
cd /mnt/nvme/GIT/OWL-ViT_test
gunicorn -w 1 -k gthread --threads 8 --timeout 120 -b 0.0.0.0:5050 prompt_converter_llm_v2:app
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

