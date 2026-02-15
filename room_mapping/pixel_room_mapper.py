#!/usr/bin/env python3
"""
pixel_room_mapper.py — Room mapper with Depth Anything support.

Grid coords:  origin = top-left (NW),  +X = east,  +Y = south.

    ┌── North (y=0) ──┐
    │                  │
  West(x=0)        East(x=W)
    │                  │
    └── South (y=H) ──┘
"""

import numpy as np, json, math, os, glob, time, hashlib
from typing import Dict, Tuple, Optional
from pathlib import Path

BASE_PATH = str(Path(__file__).resolve().parent.parent)
DEFAULT_DISTANCE_M = 5.0


# ── Tile Manager ─────────────────────────────────────────────────────

class DynamicTileManager:
    """Tile registry with 29 perceptually-distinct colours + overlap blending."""

    PALETTE = [
        (240,240,240), (45,45,45),    (0,210,80),    (160,100,40),   # free/wall/camera/door
        (230,25,75),   (0,130,200),   (255,190,0),   (145,30,180),   # 4-7
        (245,130,48),  (70,240,240),  (240,50,230),  (210,245,60),   # 8-11
        (0,128,128),   (170,110,40),  (128,128,0),   (0,75,145),     # 12-15
        (128,0,0),     (255,215,180), (170,255,195), (230,190,255),  # 16-19
        (255,250,200), (60,180,75),   (220,190,255), (255,127,80),   # 20-23
        (0,200,160),   (188,143,143), (75,0,130),    (255,99,71),    # 24-27
        (0,191,255),   (154,205,50),  (255,20,147),  (0,255,127),    # 28-31
        (218,112,214), (127,255,0),   (255,160,122), (72,61,139),    # 32-35
        (32,178,170),  (255,69,0),    (148,103,189), (44,160,44),    # 36-39
        (214,39,40),   (255,187,120), (152,223,138), (174,199,232),  # 40-43
        (197,176,213), (196,156,148), (247,182,210), (199,199,199),  # 44-47
    ]

    FREE_SPACE, WALL, CAMERA, DOOR = 0, 1, 2, 3

    def __init__(self, existing_registry=None):
        self.overlap_parents = {}
        if existing_registry:
            self.tile_registry = existing_registry.copy()
            self.next_tile_id = max(existing_registry.values()) + 1
        else:
            self.tile_registry = {'free_space': 0, 'wall': 1, 'camera': 2, 'door': 3}
            self.next_tile_id = 4
        self.id_to_name = {v: k for k, v in self.tile_registry.items()}

    def get_color(self, tid: int) -> Tuple[int, int, int]:
        if tid < len(self.PALETTE):
            return self.PALETTE[tid]
        h = hashlib.md5(str(tid).encode()).digest()
        return (h[0], h[1], h[2])

    def get_all_colors(self) -> Dict:
        return {tid: self.get_color(tid) for tid in self.tile_registry.values()}

    def get_color_registry_hex(self) -> Dict[str, str]:
        return {n: "#{:02x}{:02x}{:02x}".format(*self.get_color(t))
                for n, t in self.tile_registry.items()}

    def get_tile_type(self, obj_class: str) -> int:
        key = obj_class.lower().strip()
        if key not in self.tile_registry:
            self.tile_registry[key] = self.next_tile_id
            self.id_to_name[self.next_tile_id] = key
            self.next_tile_id += 1
        return self.tile_registry[key]

    def get_overlap_tile_type(self, existing_id: int, new_class: str) -> int:
        existing_name = self.id_to_name.get(existing_id, "unknown")
        new_name = new_class.lower().strip()
        if existing_id in (0, 1, 2, 3):
            return self.get_tile_type(new_class)
        if existing_name == new_name or " and " in existing_name:
            return existing_id
        names = sorted([existing_name, new_name])
        new_id = self.get_tile_type(f"{names[0]} and {names[1]}")
        if new_id not in self.overlap_parents:
            self.overlap_parents[new_id] = (
                self.tile_registry.get(names[0], existing_id),
                self.tile_registry.get(names[1], 0))
        return new_id

    def get_all_tiles(self) -> Dict:
        return self.tile_registry.copy()


# ── Room Geometry ────────────────────────────────────────────────────

def extract_room_geometry(scan_data: Dict) -> Optional[Dict]:
    """Extract room dimensions: depth from avg(max_depth_col_m), width from camera FOV."""
    cam = scan_data.get("camera")
    objects = scan_data.get("objects", [])
    if not cam or not objects:
        return None

    wall_depths = [obj.get("max_depth_col_m", obj.get("depth_m", 2.0))
                   for obj in objects if obj.get("max_depth_col_m", 0) > 0.1]
    wall_depth = sum(wall_depths) / len(wall_depths) if wall_depths else 2.0
    fx = cam.get("fx", 500)
    img_w = cam.get("width", 640)
    wall_width = (img_w / fx) * wall_depth
    camera_lateral = (cam.get("cx", img_w / 2) / fx) * wall_depth

    return {"wall_depth_m": round(wall_depth, 4),
            "wall_width_m": round(wall_width, 4),
            "camera_lateral_m": round(camera_lateral, 4)}


def compute_room_config(wall_geometry: Optional[Dict],
                        camera_wall: str = "north",
                        camera_position_along_wall: Optional[float] = None) -> Dict:
    """Convert wall measurements → room dimensions + camera pose.
    Yaw: north=0, south=π, east=−π/2, west=+π/2."""
    camera_wall = camera_wall.lower().strip()
    fwd = wall_geometry["wall_depth_m"] if wall_geometry else 2.0
    lat = wall_geometry["wall_width_m"] if wall_geometry else 2.5

    if camera_wall in ("north", "south"):
        room_w, room_h = lat, fwd
    elif camera_wall in ("east", "west"):
        room_w, room_h = fwd, lat
    else:
        raise ValueError(f"camera_wall must be north/south/east/west, got '{camera_wall}'")

    if camera_position_along_wall is not None:
        pos = camera_position_along_wall
    elif camera_wall in ("north", "south"):
        pos = room_w / 2
    else:
        pos = room_h / 2

    configs = {
        "north": (pos, 0.0, 0.0),           "south": (pos, room_h, math.pi),
        "east":  (room_w, pos, -math.pi/2),  "west":  (0.0, pos, math.pi/2),
    }
    cam_x, cam_y, yaw = configs[camera_wall]
    result = {"room_width_m": round(room_w, 4), "room_height_m": round(room_h, 4),
              "camera_x_m": round(cam_x, 4), "camera_y_m": round(cam_y, 4), "yaw": yaw}
    print(f"  Camera wall: {camera_wall}  |  Room: {result['room_width_m']:.2f}x"
          f"{result['room_height_m']:.2f}m  |  Camera: ({cam_x:.2f},{cam_y:.2f}) "
          f"yaw={math.degrees(yaw):.0f}deg")
    return result


# ── Pixel Room Mapper ────────────────────────────────────────────────

class PixelRoomMapper:

    def __init__(self, mode="standalone", room_width_m=2.5, room_height_m=2.5,
                 grid_resolution=0.1, res_x=None, res_y=None,
                 existing_map_file=None, existing_json_file=None,
                 room_bbox=None, room_name="main_room",
                 camera_fov_h=100, camera_fov_v=50,
                 camera_x_m=None, camera_y_m=None):
        self.mode, self.room_name = mode, room_name
        self.camera_fov_h = math.radians(camera_fov_h)
        self.camera_fov_v = math.radians(camera_fov_v)

        # Load existing data
        existing_registry = None
        self.existing_rooms = {}
        if existing_json_file and os.path.exists(existing_json_file):
            with open(existing_json_file) as f:
                d = json.load(f)
            existing_registry = d.get("tile_registry")
            self.existing_rooms = d.get("rooms", {})

        if mode == "standalone":
            self.room_width_m, self.room_height_m = room_width_m, room_height_m
            self.res_x = res_x if res_x is not None else grid_resolution
            self.res_y = res_y if res_y is not None else grid_resolution
            self.grid_resolution = grid_resolution
            self.grid_width = int(room_width_m / self.res_x + 0.5)
            self.grid_height = int(room_height_m / self.res_y + 0.5)
            self.camera_x_m = camera_x_m if camera_x_m is not None else room_width_m / 2
            self.camera_y_m = camera_y_m if camera_y_m is not None else room_height_m / 2
            self.room_bbox = (0, 0, self.grid_width, self.grid_height)
            self.map_width, self.map_height = self.grid_width, self.grid_height
        elif mode == "existing_map":
            if not existing_map_file or not room_bbox:
                raise ValueError("existing_map mode requires map file and room bbox")
            self.existing_grid = np.loadtxt(existing_map_file, dtype=np.int8)
            self.room_bbox = room_bbox
            x1, y1, x2, y2 = room_bbox
            self.room_width_m, self.room_height_m = room_width_m, room_height_m
            self.res_x = room_width_m / (x2 - x1)
            self.res_y = room_height_m / (y2 - y1)
            self.grid_resolution = (self.res_x + self.res_y) / 2
            self.grid_width, self.grid_height = x2 - x1, y2 - y1
            self.camera_x_m = camera_x_m if camera_x_m is not None else room_width_m / 2
            self.camera_y_m = camera_y_m if camera_y_m is not None else room_height_m / 2
            self.map_height, self.map_width = self.existing_grid.shape

        print(f"Room: {self.room_width_m:.2f}x{self.room_height_m:.2f}m  "
              f"({self.grid_width}x{self.grid_height} cells)  "
              f"Camera: ({self.camera_x_m:.2f},{self.camera_y_m:.2f})m")

        self.tiles = DynamicTileManager(existing_registry)
        self.all_objects = []
        self.duplicate_overlap_threshold = 0.5  # allow same-class objects in different locations

    # ── Helpers ──

    def _clamp(self, x, y):
        return (max(0.05, min(self.room_width_m - 0.05, x)),
                max(0.05, min(self.room_height_m - 0.05, y)))

    def meters_to_grid(self, x_m, y_m):
        return (max(0, min(self.grid_width - 1, int(x_m / self.res_x))),
                max(0, min(self.grid_height - 1, int(y_m / self.res_y))))

    def camera_to_grid(self):
        def snap(val, limit, res, grid_size):
            if val <= 0: return 0
            if val >= limit: return grid_size - 1
            return max(0, min(grid_size - 1, int(val / res)))
        return snap(self.camera_x_m, self.room_width_m, self.res_x, self.grid_width), \
               snap(self.camera_y_m, self.room_height_m, self.res_y, self.grid_height)

    # ── Size & Position ──

    def estimate_object_size(self, bbox, fw, fh, depth_m, intrinsics=None):
        pw, ph = bbox[2] - bbox[0], bbox[3] - bbox[1]
        if intrinsics:
            h_m = (pw / intrinsics["fx"]) * depth_m
            v_m = (ph / intrinsics["fy"]) * depth_m
        else:
            h_m = (pw / fw) * 2 * depth_m * math.tan(self.camera_fov_h / 2)
            v_m = (ph / fh) * 2 * depth_m * math.tan(self.camera_fov_v / 2)
        return (max(0.1, min(h_m, self.room_width_m / 3)),
                max(0.1, min(v_m, self.room_height_m / 3)))

    def calculate_position(self, bbox, yaw, fw, depth_m, intrinsics=None, max_depth_col_m=None):
        cx_px = (bbox[0] + bbox[2]) / 2
        # Forward depth: proportional to object's own wall distance
        if max_depth_col_m and max_depth_col_m > 0.01:
            ratio = depth_m / max_depth_col_m
            fwd = ratio * (self.room_height_m if abs(math.cos(yaw)) > 0.5 else self.room_width_m)
        else:
            fwd = depth_m
        if intrinsics:
            lateral = ((cx_px - intrinsics["cx"]) / intrinsics["fx"]) * depth_m
            ox = self.camera_x_m - lateral * math.cos(yaw) + fwd * math.sin(yaw)
            oy = self.camera_y_m + lateral * math.sin(yaw) + fwd * math.cos(yaw)
        else:
            ang = yaw - ((cx_px / fw) - 0.5) * self.camera_fov_h
            ox = self.camera_x_m + fwd * math.cos(ang)
            oy = self.camera_y_m - fwd * math.sin(ang)
        return self._clamp(ox, oy)

    # ── Duplicate detection ──

    @staticmethod
    def _cells(bbox):
        x1, y1, x2, y2 = bbox if isinstance(bbox, (list, tuple)) else bbox["bbox"]
        return {(x, y) for y in range(y1, y2) for x in range(x1, x2)}

    def _is_duplicate(self, obj_class, new_cells):
        for ex in self.all_objects:
            if ex["type"] == obj_class:
                ec = self._cells(ex["bbox"])
                shared = new_cells & ec
                smaller = min(len(new_cells), len(ec))
                if smaller > 0 and len(shared) / smaller >= self.duplicate_overlap_threshold:
                    return True
        return False

    # ── Scan Ingestion ──

    def add_scan(self, scan_data: Dict, yaw: float = 0.0):
        cam = scan_data.get("camera", {})
        intrinsics = {"fx": cam.get("fx", 500), "fy": cam.get("fy", 500),
                      "cx": cam.get("cx", cam.get("width", 640) / 2),
                      "cy": cam.get("cy", cam.get("height", 480) / 2)}
        fw = cam.get("width", 640)
        fh = cam.get("height", 480)
        detections = scan_data.get("objects", [])

        for det in detections:
            obj_class = det.get('label', '').lower().replace('a ', '').replace('an ', '').strip()
            if not obj_class: continue

            bbox = det['bbox']
            depth_m = det.get("depth_m", DEFAULT_DISTANCE_M)
            max_depth = det.get("max_depth_col_m")
            tile_type = self.tiles.get_tile_type(obj_class)
            ox, oy = self.calculate_position(bbox, yaw, fw, depth_m, intrinsics, max_depth)
            wm, hm = self.estimate_object_size(bbox, fw, fh, depth_m, intrinsics)

            # Shift position so object stays fully inside room walls
            margin = 0.05
            ox = max(margin + wm / 2, min(self.room_width_m - margin - wm / 2, ox))
            oy = max(margin + hm / 2, min(self.room_height_m - margin - hm / 2, oy))

            gx, gy = self.meters_to_grid(ox, oy)
            wc, hc = max(1, int(wm / self.res_x)), max(1, int(hm / self.res_y))
            if abs(math.sin(yaw)) > abs(math.cos(yaw)):  # east/west → swap axes
                wc, hc = hc, wc

            x1, y1 = gx - wc // 2, gy - hc // 2
            x2, y2 = x1 + wc, y1 + hc
            if self.mode == "existing_map":
                bx, by = self.room_bbox[0], self.room_bbox[1]
                x1 += bx; y1 += by; x2 += bx; y2 += by

            cells = self._cells([x1, y1, x2, y2])
            if self._is_duplicate(obj_class, cells):
                print(f"  Skip duplicate: {obj_class}"); continue

            self.all_objects.append({
                "type": obj_class, "tile_type": tile_type,
                "bbox": [x1, y1, x2, y2], "depth_m": depth_m,
                "position_m": [round(ox, 3), round(oy, 3)],
                "size_m": [round(wm, 3), round(hm, 3)]})
            print(f"  Added: {obj_class} ({ox:.2f},{oy:.2f})m depth={depth_m:.2f}m "
                  f"size=({wm:.2f}x{hm:.2f})m")

    # ── Grid Creation ──

    def create_grid_map(self) -> np.ndarray:
        T = self.tiles
        if self.mode == "standalone":
            grid = np.full((self.grid_height, self.grid_width), T.FREE_SPACE, dtype=np.int8)
            grid[0, :] = grid[-1, :] = T.WALL
            grid[:, 0] = grid[:, -1] = T.WALL
        else:
            grid = self.existing_grid.copy()
            bx1, by1, bx2, by2 = self.room_bbox
            for y in range(by1 + 1, by2 - 1):
                for x in range(bx1 + 1, bx2 - 1):
                    if y < self.map_height and x < self.map_width:
                        grid[y, x] = T.FREE_SPACE

        cx, cy = self.camera_to_grid()
        if self.mode == "existing_map":
            cx += self.room_bbox[0]; cy += self.room_bbox[1]
        if 0 <= cx < self.map_width and 0 <= cy < self.map_height:
            grid[cy, cx] = T.CAMERA

        for obj in self.all_objects:
            x1, y1, x2, y2 = obj["bbox"]
            oc = obj["type"]
            for y in range(y1, y2):
                for x in range(x1, x2):
                    if 0 < x < self.map_width - 1 and 0 < y < self.map_height - 1:
                        t = grid[y, x]
                        if t in (T.WALL, T.CAMERA, T.DOOR): continue
                        grid[y, x] = T.get_overlap_tile_type(t, oc) if t != T.FREE_SPACE \
                                     else T.get_tile_type(oc)
        return grid

    # ── Save ──

    def save(self, json_file="data/unified_rooms.json",
             map_file="data/house_map.txt", image_file="data/house_map.png"):
        grid = self.create_grid_map()
        cx, cy = self.camera_to_grid()
        if self.mode == "existing_map":
            cx += self.room_bbox[0]; cy += self.room_bbox[1]

        rooms = self.existing_rooms.copy()
        rooms[self.room_name] = {
            "name": self.room_name,
            "camera_position": [cx, cy],
            "camera_position_m": [round(self.camera_x_m, 3), round(self.camera_y_m, 3)],
            "room_dimensions_m": [round(self.room_width_m, 3), round(self.room_height_m, 3)],
            "bbox": list(self.room_bbox), "objects": self.all_objects, "doors": [25, 7]}

        output = {
            "house_dimensions_m": {"width": self.map_width * self.grid_resolution,
                                   "height": self.map_height * self.grid_resolution},
            "grid_resolution": self.grid_resolution, "rooms": rooms,
            "tile_registry": self.tiles.get_all_tiles(),
            "tile_colors": self.tiles.get_color_registry_hex()}

        with open(json_file, 'w') as f:
            json.dump(output, f, indent=2)
        np.savetxt(map_file, grid, fmt='%d')
        if image_file:
            self.save_image(grid, image_file)
        print(f"\nSaved {len(self.all_objects)} objects -> '{self.room_name}'  "
              f"({len(rooms)} rooms, {len(self.tiles.tile_registry)} tile types)")

    def save_image(self, grid, filename="data/house_map.png", cell_px=16, legend=True):
        """Render grid to PNG with colour legend."""
        from PIL import Image, ImageDraw, ImageFont
        h, w = grid.shape
        used = sorted(set(grid.flat))
        entry_h = max(cell_px, 18)
        legend_w = (entry_h + 8 + max((len(self.tiles.id_to_name.get(t, f"id_{t}"))
                    for t in used), default=6) * 9 + 20) if legend else 0

        img = Image.new("RGB", (w * cell_px + legend_w,
                         max(h * cell_px, len(used) * entry_h + 20)), (255, 255, 255))
        draw = ImageDraw.Draw(img)

        for y in range(h):
            for x in range(w):
                c = self.tiles.get_color(int(grid[y, x]))
                x0, y0 = x * cell_px, y * cell_px
                draw.rectangle([x0, y0, x0 + cell_px - 1, y0 + cell_px - 1], fill=c)

        if legend:
            try: font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
            except Exception: font = ImageFont.load_default()
            lx, ly = w * cell_px + 10, 10
            for tid in used:
                c = self.tiles.get_color(tid)
                name = self.tiles.id_to_name.get(tid, f"id_{tid}")
                draw.rectangle([lx, ly, lx + entry_h - 2, ly + entry_h - 2],
                               fill=c, outline=(0, 0, 0))
                draw.text((lx + entry_h + 4, ly + 1), name, fill=(0, 0, 0), font=font)
                ly += entry_h

        os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
        img.save(filename)
        print(f"  Image saved: {filename}")


# ── File Processing ──────────────────────────────────────────────────

def process_files(mode="standalone", existing_map=None, existing_json=None,
                  room_bbox=None, room_name="main_room", camera_wall="north",
                  camera_position_along_wall=None, grid_resolution=None,
                  grid_cells=51,
                  room_width_m=None, room_height_m=None):
    bbox_dir = os.path.join(BASE_PATH, "room_mapping/ingest_out")
    json_files = glob.glob(os.path.join(bbox_dir, "*.json"))
    if not json_files:
        print(f"No JSON files in {bbox_dir}"); return 0
    print(f"Found {len(json_files)} JSON files")

    # 1. Pre-scan for wall data
    wall_geometry = None
    for jf in sorted(json_files):
        try:
            with open(jf) as f: data = json.load(f)
            geom = extract_room_geometry(data)
            if geom:
                wall_geometry = geom
                print(f"  Wall data: depth={geom['wall_depth_m']:.2f}m "
                      f"width={geom['wall_width_m']:.2f}m"); break
        except Exception: continue

    # 2. Compute initial room config from wall geometry
    config = compute_room_config(wall_geometry, camera_wall, camera_position_along_wall)
    if room_width_m is not None:  config["room_width_m"] = room_width_m
    if room_height_m is not None: config["room_height_m"] = room_height_m

    # 3. Derive per-axis resolution from grid_cells
    #    grid_cells: int → same both axes, tuple → (along_wall, into_room)
    #    north/south: along_wall = width(X),  into_room = height(Y)
    #    east/west:   along_wall = height(Y),  into_room = width(X)
    #
    #    Resolution maps wall_depth → last INTERIOR cell (not wall cell).
    #    Grid row 0 = camera wall, row N-1 = far wall → interior = 1..(N-2).
    #    res = wall_size / (cells - 2).  Room extends slightly past wall
    #    for the boundary cells.
    if grid_resolution is None:
        if isinstance(grid_cells, (list, tuple)):
            lat_cells, depth_cells = grid_cells
        else:
            lat_cells = depth_cells = grid_cells

        if camera_wall in ("north", "south"):
            cw, ch = lat_cells, depth_cells
        else:
            cw, ch = depth_cells, lat_cells

        wall_w = config["room_width_m"]
        wall_h = config["room_height_m"]
        res_x = wall_w / max(1, cw - 2)
        res_y = wall_h / max(1, ch - 2)
        # Inflate room to include wall boundary cells
        config["room_width_m"]  = res_x * cw
        config["room_height_m"] = res_y * ch
        grid_resolution = (res_x + res_y) / 2
        print(f"  Grid: {cw}x{ch} cells (along={lat_cells}, depth={depth_cells}), "
              f"res=({res_x:.4f}, {res_y:.4f}) m/cell, "
              f"room={config['room_width_m']:.3f}x{config['room_height_m']:.3f}m")
    else:
        res_x = res_y = grid_resolution

    # Recompute camera for final room dimensions
    config = compute_room_config(
        {"wall_depth_m": config["room_height_m"] if camera_wall in ("north","south")
                         else config["room_width_m"],
         "wall_width_m": config["room_width_m"] if camera_wall in ("north","south")
                         else config["room_height_m"],
         "camera_lateral_m": 0},
        camera_wall, camera_position_along_wall)

    # 4. Create mapper
    mapper = PixelRoomMapper(
        mode=mode, room_width_m=config["room_width_m"],
        room_height_m=config["room_height_m"], grid_resolution=grid_resolution,
        res_x=res_x, res_y=res_y,
        existing_map_file=existing_map, existing_json_file=existing_json,
        room_bbox=room_bbox, room_name=room_name,
        camera_fov_h=100, camera_fov_v=50 if mode == "standalone" else 60,
        camera_x_m=config["camera_x_m"], camera_y_m=config["camera_y_m"])

    # 5. Process each file
    config_yaw = config["yaw"]
    for jf in sorted(json_files):
        try:
            print(f"Processing: {os.path.basename(jf)}")
            with open(jf) as f: sd = json.load(f)
            pose = sd.get('pose', {})
            yaw = pose['yaw'] if 'yaw' in pose else config_yaw
            mapper.add_scan(sd, yaw)
        except Exception as e:
            print(f"Error: {jf}: {e}"); import traceback; traceback.print_exc()

    mapper.save()
    return len(json_files)


# ── Main ─────────────────────────────────────────────────────────────

def main():
    # ╔═══════════════════════════════════════════════════════╗
    # ║  CONFIGURATION — edit these values for your setup    ║
    # ╚═══════════════════════════════════════════════════════╝

    mode          = "standalone"   # "standalone" or "existing_map"
    existing_map  = None           # path to existing map .txt
    existing_json = None           # path to existing .json
    room_bbox     = None           # e.g. (23, 10, 40, 24)
    room_name     = "main_room"

    #         ┌── north ──┐
    #       west        east
    #         └── south ──┘
    camera_wall = "south"          # "north" / "south" / "east" / "west"
    camera_position_along_wall = None  # metres along wall, None = middle

    grid_cells     = (51,51)            # int or (along_wall, into_room) tuple
    grid_resolution = None         # m/cell, None = auto from grid_cells
    room_width_m   = None          # override width  (m), None = auto
    room_height_m  = None          # override height (m), None = auto

    # ════════════════════════════════════════════════════════

    if room_bbox and existing_map:
        mode = "existing_map"
        print(f"Mode: Existing Map | Room: {room_name} | bbox: {room_bbox}")
    else:
        gc = f"along={grid_cells[0]} depth={grid_cells[1]}" if isinstance(grid_cells, (list, tuple)) \
             else f"{grid_cells}x{grid_cells}"
        print(f"Mode: Standalone | "
              f"Room: {room_name} | Grid: {gc}")

    print(f"Camera: {camera_wall} wall, pos={camera_position_along_wall or 'middle'}")
    print("Monitoring for detection files...  (Ctrl+C to stop)\n")

    bbox_dir = os.path.join(BASE_PATH, "room_mapping/ingest_out")
    last_count = 0
    try:
        while True:
            files = glob.glob(os.path.join(bbox_dir, "*.json"))
            if len(files) != last_count:
                if files:
                    print(f"\n[{time.strftime('%H:%M:%S')}] {len(files)} file(s)")
                    n = process_files(
                        mode, existing_map, existing_json, room_bbox, room_name,
                        camera_wall=camera_wall,
                        camera_position_along_wall=camera_position_along_wall,
                        grid_resolution=grid_resolution, grid_cells=grid_cells,
                        room_width_m=room_width_m, room_height_m=room_height_m)
                    if n: print(f"Processed {n} files -> unified_rooms.json + house_map.txt")
                else:
                    print(f"[{time.strftime('%H:%M:%S')}] No detection files")
                last_count = len(files)
            time.sleep(2)
    except KeyboardInterrupt:
        print("\n\nStopped.")


if __name__ == "__main__":
    main()