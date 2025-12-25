
# **Pipeline Stages**

on the computer:

##  *APRIL TAG**
first- get in to the docker:
```bash

 docker run -it --rm     --net=host     --ipc=host     --env="DISPLAY"     --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw"     --name ros2_apriltag     ros2-humble-apriltag
```

```bash

source src/./ros2_env.sh 
```

```bash
ros2 run apriltag_ros apriltag_node --ros-args \
  -r image_rect:=/R2/camera/image_raw \
  -r camera_info:=/R2/camera/camera_info \
  -p family:=36h11 \
  -p size:=2.00 \
  -p publish_tf:=true \
  -p qos_profile:=sensor_data \
  --log-level debug
```

in another terminal run inside the backend run the video stream:
```bash

docker exec -it backend_id bash
```
```bash
source src/./ros2_env.sh 
```
```bash
cd src/examples/src
 python3 video_stream.py 
```

in another terminal inside the backend run the service call:
```bash
docker exec -it backend_id bash
```
```bash
source src/./ros2_env.sh 
```
```bash
ros2 service call /R2/start_capture std_srvs/srv/Trigger "{}"
```


## 1. **VILA API Server**

first- get in to the jetson:
```
ssh -X user@192.168.131.22
```
**Run inside the VILA container:**
```bash
jetson-containers run -it   --publish 8080:8080   --volume /home/user/jetson-containers/data:/home/user/jetson-containers/data  nano_llm_custom /bin/bash

```

Then start the API server:
```bash
python3 -m nano_llm.chat   --api=mlc   --model Efficient-Large-Model/VILA1.5-3b   --max-context-len 256   --max-new-tokens 32   --save-json-by-image   --server --port 8080 --notify-url http://192.168.131.22:5050/from_vila
```
test:
```bash 
curl -s -X POST http://127.0.0.1:8080/describe   -H "Content-Type: application/json"   -d '{"image_path":"/mnt/VLM/jetson-data/PortraitA_01.jpg"}'
```



## 3. **NanoOWL Object Detector**
first- get in to the jetson:
```
ssh -X user@192.168.131.22
```

```bash
docker run -it --rm --name now_eng \
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
## 4. **Display Server (Web GUI Viewer)**

first- get in to the jetson:

```
ssh -X user@192.168.131.22
```
**Run:**
```bash
cd ~/GIT/NanoLLM_VILA_and_OWL
python3 python3 display_server.py  \
 --root /home/user/jetson-containers/data/R2  \
  --host 0.0.0.0   --port 8090  \
   --latest-only

```


## 5. **comm_manager.py**

first- get in to the jetson:

```
ssh -X user@192.168.131.22
```
**Run:**
```bash
cd ~/GIT/NanoLLM_VILA_and_OWL
python3 comm_manager_2.py   \
--host 0.0.0.0 \
   --port 5050  \
    --jetson2-endpoint http://192.168.131.21:5050/prompts    \
    --captures-root /home/user/jetson-containers/data/R2/ \
      --nanoowl-endpoint http://192.168.131.22:5060/infer  \ 
       --forward-timeout 45   --forward-retries 3  \
        --nanoowl-timeout 70   --nanoowl-annotate 0  
         --forward-json-url http://192.168.131.23:9090/ingest
         

 ```

## 6. **LLM Object List Extractor**

Connect to Jetson #2:
```bash
ssh user@192.168.131.21
```
in terminal 2:
```bash
cd GIT/NanoLLM_VILA_and_OWL/LLM
gunicorn -w 1 -k gthread --threads 8 --timeout 120 -b 0.0.0.0:5050 prompt_converter_llm_v2:app
```
test 
```bash
curl -s http://192.168.131.21:5050/prompts \
  -H "Content-Type: application/json" \
  -d '{"caption":"two black suitcases with red and white labels on the ground"}'
  ```

## 7. **Room Mapping + LLM Navigation Interface (Jetson #3 – 172.16.17.15)**
Connect to Jetson #3:
```bash
ssh nvidia@192.168.131.23
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


