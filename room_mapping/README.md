# House Mapping & Drone Navigation System

An autonomous drone navigation system that maps indoor environments, detects objects, and generates natural-language navigation instructions using a dual-LLM pipeline. A web GUI lets operators type tasks like "find the refrigerator" and receive both human-readable directions and step-by-step agent commands.

## Architecture

```
Camera/Drone  ──►  receiver_owl.py  ──►  pixel_room_mapper.py  ──►  render_house.py
                   (receives images       (builds grid map            (Pygame visualiser,
                    + detections)          from detections)            saves PNG for web)

Browser  ◄──►  web_mission_server_llm.py  ──►  task_request.json
                                                     │
                                    ┌────────────────┘
                                    ▼
                         llm_mission_processor.py   (LLM #1 — navigation prose)
                                    │
                                    ▼
                         mission_to_agent_commands.py (LLM #2 — agent steps)
                                    │
                                    ▼
                          agent_commands.txt  ──►  drone / agent executor
```

All components communicate through the filesystem (JSON and text files) so they can run on separate machines if needed.

## Files

**`config.json`** — Single source of truth for every tuneable parameter: file paths, server ports, LLM model names, camera geometry, grid sizing, rendering options, and polling intervals. Edit this file to change defaults for the whole system.

**`house_config.py`** — Python module that loads `config.json`, flattens it into a flat namespace, and auto-generates `--flag` CLI arguments for every key. Every other script imports `get_config()` from here.

**`run_llm_with_web.py`** — Orchestrator that launches all six components as subprocesses, monitors their health, restarts crashed services, and opens the browser. CLI arguments are forwarded to every subprocess.

**`receiver_owl.py`** — Flask endpoint (default port 9090) that accepts image metadata and detection results from the drone's vision pipeline. Saves incoming JSON to the ingest directory for the mapper to pick up.

**`pixel_room_mapper.py`** — Monitors the ingest directory for new detection files, computes object positions from camera intrinsics and depth estimates, and writes a grid map (`house_map.txt`) and room structure (`unified_rooms.json`).

**`render_house.py`** — Pygame window that renders the grid map with a colour-coded legend. Auto-reloads every 500 ms and saves a PNG snapshot (`current_map.png`) whenever the map changes, which the web server serves to the browser.

**`llm_mission_processor.py`** — First LLM stage. Watches for task requests, loads the current house JSON, and prompts the LLM to produce a natural-language navigation instruction (e.g. "walk down the hallway, MAMAD is on your right…").

**`mission_to_agent_commands.py`** — Second LLM stage. Watches for new missions and converts prose instructions into numbered agent commands (e.g. "1. Activate NavigationAgent to navigate to hallway").

**`web_mission_server_llm.py`** — Flask server (default port 8080) that serves the browser GUI (`index.html`), exposes REST endpoints for task submission and status, and bridges between the browser and the file-based LLM pipeline.

## Configuration Reference

All parameters live in `config.json` and can be overridden on the command line with `--parameter_name value`.

### Paths

| Parameter | Default | Description |
|-----------|---------|-------------|
| `unified_rooms_json` | `data/unified_rooms.json` | Output JSON containing all room structures, objects, tile registry, and colours. |
| `house_map_txt` | `data/house_map.txt` | Output text file holding the 2-D integer grid map. |
| `current_map_png` | `data/current_map.png` | PNG snapshot of the rendered map, served to the web GUI. |
| `task_request_file` | `task_request.json` | File where the web server writes incoming user tasks for the LLM. |
| `mission_response_file` | `mission_response.txt` | File where the first LLM writes its navigation prose. |
| `mission_file` | `current_mission.txt` | Copy of the mission text that the second LLM watches. |
| `agent_commands_file` | `agent_commands.txt` | File where the second LLM writes numbered agent commands. |
| `ingest_out_dir` | `ingest_out` | Directory where `receiver_owl.py` saves incoming detection JSON files. |
| `data_dir` | `data` | Parent directory for all generated map data. |

### Server

| Parameter | Default | Description |
|-----------|---------|-------------|
| `web_host` | `0.0.0.0` | Bind address for the web GUI Flask server. |
| `web_port` | `8080` | Port for the web GUI Flask server. |
| `receiver_host` | `0.0.0.0` | Bind address for the image-ingestion Flask server. |
| `receiver_port` | `9090` | Port for the image-ingestion Flask server. |

### LLM

| Parameter | Default | Description |
|-----------|---------|-------------|
| `mission_model` | `llama3.1:8b` | Ollama model used by the first LLM stage to generate navigation prose. |
| `agent_model` | `llama3.2:3b` | Ollama model used by the second LLM stage to generate agent commands. |
| `timeout` | `120` | Maximum seconds to wait for an LLM response before giving up. |

### Camera

| Parameter | Default | Description |
|-----------|---------|-------------|
| `camera_wall` | `south` | Which wall the camera faces from — one of `north`, `south`, `east`, `west`. |
| `camera_position_along_wall` | `null` | Metres along the wall where the camera sits; `null` means centred. |
| `camera_fov_h` | `100` | Horizontal field-of-view of the camera in degrees. |
| `camera_fov_v` | `50` | Vertical field-of-view of the camera in degrees. |

### Room / Grid

| Parameter | Default | Description |
|-----------|---------|-------------|
| `mode` | `standalone` | Mapping mode — `standalone` creates a new map, `existing_map` adds a room to an existing grid. |
| `room_name` | `main_room` | Name assigned to the room being mapped. |
| `grid_cells` | `[51, 51]` | Grid dimensions as `[along_wall, into_room]`; used when `grid_resolution` is null. |
| `grid_resolution` | `null` | Metres per cell; `null` means auto-compute from `grid_cells` and wall geometry. |
| `room_width_m` | `null` | Override room width in metres; `null` means auto-detect from depth data. |
| `room_height_m` | `null` | Override room height in metres; `null` means auto-detect from depth data. |
| `room_bbox` | `null` | Bounding box `[x1, y1, x2, y2]` in an existing grid for `existing_map` mode. |
| `existing_map` | `null` | Path to an existing `house_map.txt` when using `existing_map` mode. |
| `existing_json` | `null` | Path to an existing `unified_rooms.json` when using `existing_map` mode. |
| `default_distance_m` | `5.0` | Fallback depth estimate in metres when a detection has no depth value. |

### Rendering

| Parameter | Default | Description |
|-----------|---------|-------------|
| `cell_size` | `25` | Pixel size of each grid cell in the Pygame renderer (range 5–50). |
| `legend_width` | `400` | Pixel width of the object-legend sidebar in the Pygame window. |
| `reload_interval` | `500` | Milliseconds between auto-reload checks in the renderer. |

### Receiver

| Parameter | Default | Description |
|-----------|---------|-------------|
| `accumulate_mode` | `false` | When true, keeps all ingested files; when false, clears old files on each new image. |

### Polling Intervals

| Parameter | Default | Description |
|-----------|---------|-------------|
| `llm_poll_interval` | `0.5` | Seconds between checks for new task requests or missions in the LLM loops. |
| `web_poll_interval` | `0.25` | Seconds between checks while the web server waits for LLM responses. |
| `data_update_interval` | `1` | Seconds between background reloads of house data in the web server. |
| `mapper_poll_interval` | `2` | Seconds between scans for new detection files in the mapper. |

## Prerequisites

- Python 3.8+
- [Ollama](https://ollama.com/) with the models you configure (defaults: `llama3.1:8b` and `llama3.2:3b`)
- Python packages: `flask`, `flask-cors`, `numpy`, `pygame`
- An `index.html` file in the working directory for the web GUI

Install dependencies:

```bash
pip install flask flask-cors numpy pygame
ollama pull llama3.1:8b
ollama pull llama3.2:3b
```

## Quick Start

```bash
# Launch everything with defaults from config.json
python run_llm_with_web.py
```

The orchestrator starts all six components, opens `http://localhost:8080` in your browser, and prints status. Press `Ctrl+C` to stop everything.

## CLI Overrides

Every `config.json` parameter becomes a `--flag`:

```bash
# Change web port and camera orientation
python run_llm_with_web.py --web_port 9999 --camera_wall north

# Use larger models
python run_llm_with_web.py --mission_model llama3.1:70b --agent_model llama3.1:8b

# Adjust grid size and cell rendering
python run_llm_with_web.py --grid_cells 80 80 --cell_size 15

# Use a different config file entirely
python run_llm_with_web.py --config production_config.json

# Use office map
python run_llm_with_web.py --mode existing_map --room_bbox 23 10 40 24 --room_name MAMAD --existing_map office_map.txt --existing_json office.json --camera_wall south
```

Individual scripts accept the same flags:

```bash
python pixel_room_mapper.py --camera_wall west --grid_cells 30 40
python receiver_owl.py --receiver_port 9191 --accumulate_mode true
python render_house.py --cell_size 10 --legend_width 300
```

## Running Components Separately

For distributed setups (e.g. LLM on a GPU server, web on a laptop), run each script individually. They communicate through shared files, so mount or sync the working directory.

```bash
# Machine A: vision pipeline
python receiver_owl.py
python pixel_room_mapper.py

# Machine B: LLM processing
python llm_mission_processor.py
python mission_to_agent_commands.py

# Machine C: user interface
python web_mission_server_llm.py
python render_house.py
```

## File Exchange Protocol

The components are decoupled through files:

1. **Drone → `receiver_owl.py`**: HTTP POST with detection metadata to `/ingest`
2. **`receiver_owl.py` → `pixel_room_mapper.py`**: JSON files in `ingest_out/`
3. **`pixel_room_mapper.py` → `render_house.py` / `web_mission_server_llm.py`**: `data/unified_rooms.json` + `data/house_map.txt`
4. **Browser → `web_mission_server_llm.py`**: POST to `/api/generate` with `{"task": "..."}`
5. **`web_mission_server_llm.py` → `llm_mission_processor.py`**: `task_request.json`
6. **`llm_mission_processor.py` → `mission_to_agent_commands.py`**: `current_mission.txt`
7. **`mission_to_agent_commands.py` → Browser**: `agent_commands.txt` (polled by web server)