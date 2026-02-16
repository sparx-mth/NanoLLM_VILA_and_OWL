
# **Pipeline Stages**

## 1. ** VLLM WITH QWEN:**

first- get in to the jetson:
```
ssh -X user@192.168.131.22
```

in terminal 1:
```bash
docker run --rm -it \
  --runtime nvidia \
  --network host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm_qwen3_vl_4b_instruct_aws_4bit:latest
```


Inside the container, start the vLLM API server:

```
vllm serve cpatonn/Qwen3-VL-4B-Instruct-AWQ-4bit \
  --host 0.0.0.0 \
  --port 8080 \
  --dtype float16 \
  --gpu-memory-utilization 0.5 \
  --max-model-len 2048 \
  --max-num-batched-tokens 128 \
  --max-num-seqs 4 \
  --swap-space 0 \
  --enforce-eager
```

** if its not working, try:
  --max-model-len 512 
  --max-num-seqs 1
  --gpu-memory-utilization 0.4


in terminal 2:
```
cd /jetson-containers/data
python3 -m http.server 9000 --bind 0.0.0.0
```

test:
```
curl -s http://127.0.0.1:8080/v1/chat/completions   -H "Content-Type: application/json"   -d '{
    "model": "cpatonn/Qwen3-VL-4B-Instruct-AWQ-4bit",
    "messages": [{
      "role": "user",
      "content": [
        {"type":"text","text":"Describe in a short list JUST the objects in the image."},
        {"type":"image_url","image_url":{"url":"http://172.16.17.15:9000/R1/latest/R1_20260127_133755.jpg"}}
      ]
    }],
    "max_tokens": 64
  }' | jq -r '.choices[0].message.content'
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
python3 display_server.py  \
 --root /home/user/jetson-containers/data/R1  \
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
python3 comm_manager_vllm.py --profile robotican   --vllm-model espressor/meta-llama.Llama-3.2-3B-Instruct_W4A16   --captures-root /home/user/jetson-containers/data/R1/   --endpoint http://192.168.131.22:8080  --host 192.168.131.22  --force --depth-endpoint http://192.168.131.22:5070/bbox_depth



```
## 6. **capture_on_enter.py**
first- get in to the jetson:

```
ssh -X user@192.168.131.22
```
**Run:**
```bash
cd ~/GIT/NanoLLM_VILA_and_OWL
python3 capture_on_enter.py --rows 1 --cols 1 --overlap 0.0
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
cd ~/GIT/NanoLLM_VILA_and_OWL/room_mapping
source .venv/bin/activate
pip3 install requirements.txt
python3 run_llm_with_web.py
```

## 8. Depth Anythnig v3 
**Build (colcon)**
From your workspace root:
```bash
cd ~/depth_anything_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select depth_anything_v3 --cmake-args -DCMAKE_BUILD_TYPE=Release
source install/setup.bash
```
**Run the HTTP server**
```bash
ros2 run depth_anything_v3 depth_anything_http_server    --model /home/user/depth_anything_ws/src/ros2-depth-anything-v3-trt/onnx/DA3-SMALL/DA3-SMALL.fp16-batch1.engine   --camera-yaml /home/user/depth_anything_ws/src/ros2-depth-anything-v3-trt/camera_info_4k.yaml   --max-depth 7.0 --host 0.0.0.0 --port 5070 --save-depth --tiling 1 --tile-overlap-div 2 --tile-fade-px 400

```

### Notes:
- Saving happens **per request** (not per second). `--save-every N` saves every N-th request.
- The save directory is derived from the request's `image_dir`:
  if `image_dir=/a/b/c` then depth images go under `/a/b/c_depth/`.



### old

 - if you need llm jetson nano - 
in terminal 1:
```bash
docker run --rm -it \
  --runtime nvidia \
  --network host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm_llama_3b:latest
```


Inside the container, start the vLLM API server:

```
vllm serve espressor/meta-llama.Llama-3.2-3B-Instruct_W4A16 \
  --dtype float16 \
  --gpu-memory-utilization 0.5 \
  --max-model-len 512 \
  --max-num-batched-tokens 128 \
  --max-num-seqs 1 \
  --swap-space 0 \
  --enforce-eager
```

***no need***
Just if you want to check the llm:
in terminal 2:
```bash
cd GIT/NanoLLM_VILA_and_OWL/LLM
python3 prompt_converter_vllm.py
```
test 
```bash


curl -s http://192.168.131.21:5050/prompts \
  -H "Content-Type: application/json" \
  -d '{"caption":"two black suitcases with red and white labels on the ground"}'
  ```