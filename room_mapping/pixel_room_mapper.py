#!/usr/bin/env python3
"""
pixel_room_mapper.py - Dual-mode room mapper with Depth Anything support

Features:
  - Auto room dimensions from wall depth measurements (first JSON file)
  - Easy camera_wall selector: "north" / "south" / "east" / "west"
  - Per-object depth from Depth Anything (falls back to DEFAULT_DISTANCE_M)
  - Both old (nanoowl/pose) and new (Depth Anything) JSON formats

Room coordinate system (grid):
    Origin = top-left corner (northwest)
    +X  = east  (right in grid)
    +Y  = south (down in grid)

    ┌──── North wall (y=0) ────┐
    │                          │
  West                       East
  (x=0)                    (x=W)
    │                          │
    └──── South wall (y=H) ────┘
"""

import numpy as np
import json
import math
import os
import glob
import time
from typing import Dict, List, Tuple, Optional
from pathlib import Path

BASE_PATH = str(Path(__file__).resolve().parent.parent)

# Default depth when Depth Anything data is not available
DEFAULT_DISTANCE_M = 5.0


# ======================================================================
# Tile manager
# ======================================================================

class DynamicTileManager:
    """Manages dynamic tile types."""

    def __init__(self, existing_registry=None):
        self.FREE_SPACE = 0
        self.WALL = 1
        self.CAMERA = 2
        self.DOOR = 3

        if existing_registry:
            self.tile_registry = existing_registry.copy()
            self.next_tile_id = max(existing_registry.values()) + 1
        else:
            self.tile_registry = {
                'free_space': self.FREE_SPACE,
                'wall': self.WALL,
                'camera': self.CAMERA,
                'door': self.DOOR
            }
            self.next_tile_id = 4

        self.id_to_name = {v: k for k, v in self.tile_registry.items()}

    def get_tile_type(self, object_class: str) -> int:
        obj_key = object_class.lower().strip()
        if obj_key not in self.tile_registry:
            self.tile_registry[obj_key] = self.next_tile_id
            self.id_to_name[self.next_tile_id] = obj_key
            self.next_tile_id += 1
        return self.tile_registry[obj_key]

    def get_overlap_tile_type(self, existing_tile_id: int, new_object_class: str) -> int:
        existing_name = self.id_to_name.get(existing_tile_id, "unknown")
        new_name = new_object_class.lower().strip()

        if existing_tile_id in [self.FREE_SPACE, self.WALL, self.CAMERA, self.DOOR]:
            return self.get_tile_type(new_object_class)
        if existing_name == new_name:
            return existing_tile_id
        if " and " in existing_name:
            return existing_tile_id

        names = sorted([existing_name, new_name])
        return self.get_tile_type(f"{names[0]} and {names[1]}")

    def get_all_tiles(self) -> Dict:
        return self.tile_registry.copy()


# ======================================================================
# Wall geometry extraction  (camera-frame measurements)
# ======================================================================

def extract_room_geometry(scan_data: Dict) -> Optional[Dict]:
    """
    Extract room measurements from a Depth Anything scan with ``wall`` data.

    Returns camera-frame measurements (orientation-agnostic)::

        {
            "wall_depth_m":     float,  # distance to opposite wall
            "wall_width_m":     float,  # lateral span of opposite wall
            "camera_lateral_m": float,  # camera offset from left-of-image edge
        }

    or ``None`` if the scan has no usable wall data.
    """
    if "wall" not in scan_data or "camera" not in scan_data:
        return None

    wall = scan_data["wall"]
    cam = scan_data["camera"]

    for key in ("left", "right"):
        if key not in wall or not wall[key].get("valid", False):
            return None

    fx = cam.get("fx", 500)
    cx = cam.get("cx", cam.get("width", 640) / 2)

    # Average depth of all valid samples
    depths = []
    for key in ("left", "middle", "right"):
        entry = wall.get(key)
        if entry and entry.get("valid", False):
            depths.append(entry["depth_m"])
    wall_depth = sum(depths) / len(depths)

    # Project outer bbox edges to metres at wall depth (pinhole model)
    left_pixel = wall["left"]["bbox"][0]
    right_pixel = wall["right"]["bbox"][2]

    x_left_m = ((left_pixel - cx) / fx) * wall_depth
    x_right_m = ((right_pixel - cx) / fx) * wall_depth

    wall_width = x_right_m - x_left_m
    camera_lateral = -x_left_m

    return {
        "wall_depth_m": round(wall_depth, 4),
        "wall_width_m": round(wall_width, 4),
        "camera_lateral_m": round(camera_lateral, 4),
    }


# ======================================================================
# Wall → room coordinate mapping
# ======================================================================

def compute_room_config(wall_geometry: Optional[Dict],
                        camera_wall: str = "north",
                        camera_position_along_wall: Optional[float] = None
                        ) -> Dict:
    """
    Convert camera-frame wall measurements to a full room configuration.

    Args:
        wall_geometry: Output of ``extract_room_geometry``, or None.
        camera_wall:   ``"north"`` / ``"south"`` / ``"east"`` / ``"west"``
        camera_position_along_wall:
            Position in metres along the wall (room coords):
              - North / South → distance from **west** wall   (room X)
              - East  / West  → distance from **north** wall  (room Y)
            ``None`` → **middle of the wall**.

    Returns:
        dict with ``room_width_m``, ``room_height_m``,
        ``camera_x_m``, ``camera_y_m``, ``yaw``.

    Yaw conventions:

        Wall    yaw       Camera looks → room
        ─────   ────────  ─────────────────────
        north    0        +Y  (south)
        south    π        −Y  (north)
        east    −π/2      −X  (west)
        west    +π/2      +X  (east)
    """
    camera_wall = camera_wall.lower().strip()

    if wall_geometry is not None:
        fwd = wall_geometry["wall_depth_m"]
        lat = wall_geometry["wall_width_m"]
    else:
        fwd = 2.0
        lat = 2.5

    # Room dimensions depend on which wall the camera is on
    if camera_wall in ("north", "south"):
        room_w, room_h = lat, fwd
    elif camera_wall in ("east", "west"):
        room_w, room_h = fwd, lat
    else:
        raise ValueError(f"camera_wall must be north/south/east/west, got '{camera_wall}'")

    # Position along the wall: default = middle
    if camera_position_along_wall is not None:
        pos = camera_position_along_wall
    elif camera_wall in ("north", "south"):
        pos = room_w / 2
    else:
        pos = room_h / 2

    # Camera position and yaw per wall
    configs = {
        "north": (pos,    0.0,    0.0),
        "south": (pos,    room_h, math.pi),
        "east":  (room_w, pos,    -math.pi / 2),
        "west":  (0.0,    pos,    math.pi / 2),
    }
    cam_x, cam_y, yaw = configs[camera_wall]

    result = {
        "room_width_m": round(room_w, 4),
        "room_height_m": round(room_h, 4),
        "camera_x_m": round(cam_x, 4),
        "camera_y_m": round(cam_y, 4),
        "yaw": yaw,
    }

    print(f"  Camera wall: {camera_wall}")
    print(f"  Room: {result['room_width_m']:.2f} x {result['room_height_m']:.2f} m")
    print(f"  Camera: ({result['camera_x_m']:.2f}, {result['camera_y_m']:.2f}) m, "
          f"yaw={math.degrees(yaw):.0f}°")

    return result


# ======================================================================
# PixelRoomMapper
# ======================================================================

class PixelRoomMapper:
    """Room mapper with Depth Anything support and configurable camera wall."""

    def __init__(self,
                 mode: str = "standalone",
                 room_width_m: float = 2.5,
                 room_height_m: float = 2.5,
                 grid_resolution: float = 0.1,
                 existing_map_file: Optional[str] = None,
                 existing_json_file: Optional[str] = None,
                 room_bbox: Optional[Tuple[int, int, int, int]] = None,
                 room_name: str = "main_room",
                 camera_fov_h: float = 100,
                 camera_fov_v: float = 50,
                 camera_x_m: Optional[float] = None,
                 camera_y_m: Optional[float] = None):

        self.mode = mode
        self.room_name = room_name
        self.camera_fov_h = math.radians(camera_fov_h)
        self.camera_fov_v = math.radians(camera_fov_v)

        # Load existing JSON
        existing_registry = None
        self.existing_rooms = {}
        if existing_json_file and os.path.exists(existing_json_file):
            with open(existing_json_file, 'r') as f:
                existing_data = json.load(f)
                existing_registry = existing_data.get("tile_registry", None)
                self.existing_rooms = existing_data.get("rooms", {})

        if mode == "standalone":
            self.room_width_m = room_width_m
            self.room_height_m = room_height_m
            self.grid_resolution = grid_resolution
            self.grid_width = int(room_width_m / grid_resolution)
            self.grid_height = int(room_height_m / grid_resolution)
            self.camera_x_m = camera_x_m if camera_x_m is not None else room_width_m / 2
            self.camera_y_m = camera_y_m if camera_y_m is not None else room_height_m / 2
            self.room_bbox = (0, 0, self.grid_width, self.grid_height)
            self.map_width = self.grid_width
            self.map_height = self.grid_height

        elif mode == "existing_map":
            if not existing_map_file or not room_bbox:
                raise ValueError("existing_map mode requires map file and room bbox")
            self.existing_grid = np.loadtxt(existing_map_file, dtype=np.int8)
            self.room_bbox = room_bbox
            x1, y1, x2, y2 = room_bbox
            room_width_cells = x2 - x1
            room_height_cells = y2 - y1
            self.room_width_m = room_width_m
            self.room_height_m = room_height_m
            # Separate resolution per axis for accurate mapping
            self.res_x = self.room_width_m / room_width_cells
            self.res_y = self.room_height_m / room_height_cells
            self.grid_resolution = (self.res_x + self.res_y) / 2
            self.grid_width = room_width_cells
            self.grid_height = room_height_cells
            self.camera_x_m = camera_x_m if camera_x_m is not None else self.room_width_m / 2
            self.camera_y_m = camera_y_m if camera_y_m is not None else self.room_height_m / 2
            self.map_height, self.map_width = self.existing_grid.shape

        # For standalone, res_x == res_y == grid_resolution
        if mode == "standalone":
            self.res_x = self.grid_resolution
            self.res_y = self.grid_resolution

        print(f"Room: {self.room_width_m:.2f} x {self.room_height_m:.2f} m  "
              f"({self.grid_width} x {self.grid_height} cells, "
              f"res=({self.res_x:.4f}, {self.res_y:.4f}) m/cell)")
        print(f"Camera: ({self.camera_x_m:.2f}, {self.camera_y_m:.2f}) m")

        self.tiles = DynamicTileManager(existing_registry)
        self.all_objects = []
        self.duplicate_overlap_threshold = 0.0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_depth_anything_format(scan_data: Dict) -> bool:
        return "camera" in scan_data and "objects" in scan_data

    @staticmethod
    def _get_object_depth(detection: Dict) -> float:
        return detection.get("depth_m", DEFAULT_DISTANCE_M)

    # ------------------------------------------------------------------
    # Size estimation
    # ------------------------------------------------------------------

    def estimate_object_size_from_pixels(self, bbox, frame_width, frame_height,
                                         depth_m, object_class="",
                                         camera_intrinsics=None):
        """Pinhole model (with intrinsics) or FOV fallback."""
        pixel_w = bbox[2] - bbox[0]
        pixel_h = bbox[3] - bbox[1]

        if camera_intrinsics is not None:
            h_m = (pixel_w / camera_intrinsics["fx"]) * depth_m
            v_m = (pixel_h / camera_intrinsics["fy"]) * depth_m
        else:
            h_m = (pixel_w / frame_width) * 2 * depth_m * math.tan(self.camera_fov_h / 2)
            v_m = (pixel_h / frame_height) * 2 * depth_m * math.tan(self.camera_fov_v / 2)

        h_m = max(0.1, min(h_m, self.room_width_m / 3))
        v_m = max(0.1, min(v_m, self.room_height_m / 3))
        return h_m, v_m

    # ------------------------------------------------------------------
    # Position estimation
    # ------------------------------------------------------------------

    def calculate_object_position(self, bbox, yaw, frame_width, depth_m,
                                  camera_intrinsics=None):
        """
        Project object into room coordinates.

        Top-down projection mirrors the camera X axis relative to a naive
        2D rotation, so the lateral component is negated in the transform:

            room_dx = -lateral · cos(yaw)  +  forward · sin(yaw)
            room_dy =  lateral · sin(yaw)  +  forward · cos(yaw)

        This correctly maps:
            north (yaw=0):    right-in-image → west  (−X)
            south (yaw=π):    right-in-image → east  (+X)
            east  (yaw=−π/2): right-in-image → north (−Y)
            west  (yaw=+π/2): right-in-image → south (+Y)
        """
        bbox_center_x = (bbox[0] + bbox[2]) / 2

        if camera_intrinsics is not None:
            cx = camera_intrinsics["cx"]
            fx = camera_intrinsics["fx"]
            lateral_m = ((bbox_center_x - cx) / fx) * depth_m
            forward_m = depth_m
        else:
            # Legacy FOV mode (uses its own yaw convention)
            angle_offset = -((bbox_center_x / frame_width) - 0.5) * self.camera_fov_h
            object_angle = yaw + angle_offset
            obj_x = self.camera_x_m + depth_m * math.cos(object_angle)
            obj_y = self.camera_y_m - depth_m * math.sin(object_angle)
            obj_x = max(0.05, min(self.room_width_m - 0.05, obj_x))
            obj_y = max(0.05, min(self.room_height_m - 0.05, obj_y))
            return obj_x, obj_y

        # FIX #1: negate lateral in the rotation to account for
        # the mirror between 3D camera view and 2D top-down map
        obj_x = self.camera_x_m - lateral_m * math.cos(yaw) + forward_m * math.sin(yaw)
        obj_y = self.camera_y_m + lateral_m * math.sin(yaw) + forward_m * math.cos(yaw)

        obj_x = max(0.05, min(self.room_width_m - 0.05, obj_x))
        obj_y = max(0.05, min(self.room_height_m - 0.05, obj_y))
        return obj_x, obj_y

    # ------------------------------------------------------------------
    # Grid helpers
    # ------------------------------------------------------------------

    def meters_to_grid(self, x_m: float, y_m: float) -> Tuple[int, int]:
        """Convert metres to grid coordinates using per-axis resolution."""
        gx = int(x_m / self.res_x)
        gy = int(y_m / self.res_y)
        gx = max(0, min(self.grid_width - 1, gx))
        gy = max(0, min(self.grid_height - 1, gy))
        return gx, gy

    def camera_to_grid(self) -> Tuple[int, int]:
        """
        Convert camera metres to grid cell, snapping to wall boundaries.
        FIX #3: guarantees the camera lands exactly on the wall cell.
        """
        # Snap X
        if self.camera_x_m <= 0:
            gx = 0
        elif self.camera_x_m >= self.room_width_m:
            gx = self.grid_width - 1
        else:
            gx = int(self.camera_x_m / self.res_x)
            gx = max(0, min(self.grid_width - 1, gx))

        # Snap Y
        if self.camera_y_m <= 0:
            gy = 0
        elif self.camera_y_m >= self.room_height_m:
            gy = self.grid_height - 1
        else:
            gy = int(self.camera_y_m / self.res_y)
            gy = max(0, min(self.grid_height - 1, gy))

        return gx, gy

    def get_object_cells(self, obj):
        x1, y1, x2, y2 = obj["bbox"]
        return {(x, y) for y in range(y1, y2) for x in range(x1, x2)}

    def is_duplicate_object(self, obj_class, new_cells):
        for existing in self.all_objects:
            if existing["type"] == obj_class:
                existing_cells = self.get_object_cells(existing)
                shared = new_cells & existing_cells
                smaller = min(len(new_cells), len(existing_cells))
                if smaller > 0 and len(shared) / smaller >= self.duplicate_overlap_threshold:
                    return True
        return False

    # ------------------------------------------------------------------
    # Scan ingestion
    # ------------------------------------------------------------------

    def add_scan(self, scan_data: Dict, yaw: float = 0.0):
        """Add a scan.  ``yaw`` is the camera heading in room coordinates."""
        camera_intrinsics = None

        if self._is_depth_anything_format(scan_data):
            cam = scan_data["camera"]
            camera_intrinsics = {
                "fx": cam.get("fx", 500), "fy": cam.get("fy", 500),
                "cx": cam.get("cx", cam.get("width", 640) / 2),
                "cy": cam.get("cy", cam.get("height", 480) / 2),
            }
            frame_width = cam.get("width", 640)
            frame_height = cam.get("height", 480)
            detections = scan_data.get("objects", [])

        elif 'nanoowl' in scan_data and 'result' in scan_data['nanoowl']:
            result = scan_data['nanoowl']['result']
            img = result.get('image', {})
            frame_width = img.get('width', 1280)
            frame_height = img.get('height', 720)
            detections = result.get('detections', [])

        elif 'image' in scan_data:
            frame_width = scan_data['image'].get('width', 1280)
            frame_height = scan_data['image'].get('height', 720)
            detections = scan_data.get('detections', [])
        else:
            frame_width, frame_height = 1280, 720
            detections = scan_data.get('detections', [])

        for det in detections:
            label = det.get('label', '').lower()
            obj_class = label.replace('a ', '').replace('an ', '').strip()
            if not obj_class:
                continue

            bbox = det['bbox']
            depth_m = self._get_object_depth(det)
            tile_type = self.tiles.get_tile_type(obj_class)

            obj_x, obj_y = self.calculate_object_position(
                bbox, yaw, frame_width, depth_m, camera_intrinsics)

            width_m, height_m = self.estimate_object_size_from_pixels(
                bbox, frame_width, frame_height, depth_m, obj_class, camera_intrinsics)

            # Constrain size to stay inside room
            max_w = max(0.1, min(obj_x - 0.05, self.room_width_m - obj_x - 0.05) * 2)
            max_h = max(0.1, min(obj_y - 0.05, self.room_height_m - obj_y - 0.05) * 2)
            width_m = min(width_m, max_w)
            height_m = min(height_m, max_h)

            # Grid coords using per-axis resolution
            gx, gy = self.meters_to_grid(obj_x, obj_y)
            wc = max(1, int(width_m / self.res_x))
            hc = max(1, int(height_m / self.res_y))

            # FIX #2: swap when camera faces along X axis (east/west walls)
            # width_m  = lateral extent (camera horizontal) → along the wall
            # height_m = vertical extent (we use as depth on floor) → into the room
            # When yaw ≈ ±π/2 the "along wall" axis is Y in the grid, so swap.
            if abs(math.sin(yaw)) > abs(math.cos(yaw)):
                wc, hc = hc, wc

            x1 = gx - wc // 2
            y1 = gy - hc // 2
            x2, y2 = x1 + wc, y1 + hc

            if self.mode == "existing_map":
                x1 += self.room_bbox[0]; y1 += self.room_bbox[1]
                x2 += self.room_bbox[0]; y2 += self.room_bbox[1]

            new_cells = self.get_object_cells({"bbox": [x1, y1, x2, y2]})
            if self.is_duplicate_object(obj_class, new_cells):
                print(f"  Skipping duplicate: {obj_class}")
                continue

            self.all_objects.append({
                "type": obj_class, "tile_type": tile_type,
                "bbox": [x1, y1, x2, y2], "depth_m": depth_m,
                "position_m": [round(obj_x, 3), round(obj_y, 3)],
                "size_m": [round(width_m, 3), round(height_m, 3)],
            })
            print(f"  Added: {obj_class} at ({obj_x:.2f}, {obj_y:.2f}) m, "
                  f"depth={depth_m:.2f} m, size=({width_m:.2f} x {height_m:.2f}) m")

    # ------------------------------------------------------------------
    # Grid creation
    # ------------------------------------------------------------------

    def create_grid_map(self) -> np.ndarray:
        if self.mode == "standalone":
            grid = np.full((self.grid_height, self.grid_width),
                           self.tiles.FREE_SPACE, dtype=np.int8)
            for x in range(self.grid_width):
                grid[0, x] = grid[self.grid_height - 1, x] = self.tiles.WALL
            for y in range(self.grid_height):
                grid[y, 0] = grid[y, self.grid_width - 1] = self.tiles.WALL
        else:
            grid = self.existing_grid.copy()
            bx1, by1, bx2, by2 = self.room_bbox
            for y in range(by1 + 1, by2 - 1):
                for x in range(bx1 + 1, bx2 - 1):
                    if y < self.map_height and x < self.map_width:
                        grid[y, x] = self.tiles.FREE_SPACE

        # FIX #3: use camera_to_grid for wall snapping
        cx, cy = self.camera_to_grid()
        if self.mode == "existing_map":
            cx += self.room_bbox[0]
            cy += self.room_bbox[1]
        if 0 <= cx < self.map_width and 0 <= cy < self.map_height:
            grid[cy, cx] = self.tiles.CAMERA

        # Objects
        for obj in self.all_objects:
            x1, y1, x2, y2 = obj["bbox"]
            obj_class = obj["type"]
            for y in range(y1, y2):
                for x in range(x1, x2):
                    if 0 < x < self.map_width - 1 and 0 < y < self.map_height - 1:
                        t = grid[y, x]
                        if t in (self.tiles.WALL, self.tiles.CAMERA, self.tiles.DOOR):
                            continue
                        if t != self.tiles.FREE_SPACE:
                            grid[y, x] = self.tiles.get_overlap_tile_type(t, obj_class)
                        else:
                            grid[y, x] = self.tiles.get_tile_type(obj_class)
        return grid

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save(self, json_file="data/unified_rooms.json",
             map_file="data/house_map.txt"):
        grid = self.create_grid_map()

        cx, cy = self.camera_to_grid()
        if self.mode == "existing_map":
            cx += self.room_bbox[0]
            cy += self.room_bbox[1]

        rooms = self.existing_rooms.copy()
        rooms[self.room_name] = {
            "name": self.room_name,
            "camera_position": [cx, cy],
            "camera_position_m": [round(self.camera_x_m, 3),
                                  round(self.camera_y_m, 3)],
            "room_dimensions_m": [round(self.room_width_m, 3),
                                  round(self.room_height_m, 3)],
            "bbox": list(self.room_bbox),
            "objects": self.all_objects,
            "doors": [25, 7],
        }

        output = {
            "house_dimensions_m": {
                "width": self.map_width * self.grid_resolution,
                "height": self.map_height * self.grid_resolution,
            },
            "grid_resolution": self.grid_resolution,
            "rooms": rooms,
            "tile_registry": self.tiles.get_all_tiles(),
        }

        with open(json_file, 'w') as f:
            json.dump(output, f, indent=2)
        np.savetxt(map_file, grid, fmt='%d')

        print(f"\nSaved {len(self.all_objects)} objects to room '{self.room_name}'")
        print(f"Total rooms: {len(rooms)}")
        print(f"Tile types: {len(self.tiles.tile_registry)} total")


# ======================================================================
# File processing
# ======================================================================

def get_yaw_from_json(scan_data: Dict) -> Optional[float]:
    """Extract yaw from legacy pose field.  Returns None if absent."""
    if 'pose' in scan_data and 'yaw' in scan_data['pose']:
        yaw = scan_data['pose']['yaw']
        print(f"  Found yaw in JSON: {yaw:.4f} rad ({math.degrees(yaw):.1f}°)")
        return yaw
    return None


def process_files(mode="standalone", existing_map=None, existing_json=None,
                  room_bbox=None, room_name="main_room",
                  camera_wall="north",
                  camera_position_along_wall=None,
                  grid_resolution=0.05):
    """
    Process all detection files.

    1. Pre-scan for first JSON with ``wall`` data → auto room dimensions.
    2. ``compute_room_config`` maps wall + camera_wall → room coords + yaw.
    3. Each file is processed with the configured yaw (Depth Anything) or
       per-file pose yaw (legacy).
    """
    bbox_dir = os.path.join(BASE_PATH, "room_mapping/ingest_out")
    json_files = glob.glob(os.path.join(bbox_dir, "*.json"))

    if not json_files:
        print(f"No JSON files found in {bbox_dir}")
        return 0

    print(f"Found {len(json_files)} JSON files to process")

    # ------------------------------------------------------------------
    # 1. Pre-scan for wall data
    # ------------------------------------------------------------------
    wall_geometry = None
    for jf in sorted(json_files):
        try:
            with open(jf, 'r') as f:
                data = json.load(f)
            geom = extract_room_geometry(data)
            if geom is not None:
                wall_geometry = geom
                print(f"  Wall data from {os.path.basename(jf)}:")
                print(f"    depth={geom['wall_depth_m']:.2f} m, "
                      f"width={geom['wall_width_m']:.2f} m, "
                      f"lateral={geom['camera_lateral_m']:.2f} m")
                break
        except Exception:
            continue

    # ------------------------------------------------------------------
    # 2. Compute room config (works with or without wall data)
    # ------------------------------------------------------------------
    config = compute_room_config(wall_geometry, camera_wall,
                                 camera_position_along_wall)
    config_yaw = config["yaw"]

    # ------------------------------------------------------------------
    # 3. Create mapper
    # ------------------------------------------------------------------
    if mode == "standalone":
        mapper = PixelRoomMapper(
            mode="standalone",
            room_width_m=config["room_width_m"],
            room_height_m=config["room_height_m"],
            grid_resolution=grid_resolution,
            existing_json_file=existing_json,
            room_name=room_name,
            camera_fov_h=100, camera_fov_v=50,
            camera_x_m=config["camera_x_m"],
            camera_y_m=config["camera_y_m"],
        )
    else:
        mapper = PixelRoomMapper(
            mode="existing_map",
            room_width_m=config["room_width_m"],
            room_height_m=config["room_height_m"],
            existing_map_file=existing_map,
            existing_json_file=existing_json,
            room_bbox=room_bbox,
            room_name=room_name,
            camera_fov_h=100, camera_fov_v=60,
            camera_x_m=config["camera_x_m"],
            camera_y_m=config["camera_y_m"],
        )

    # ------------------------------------------------------------------
    # 4. Process each file
    # ------------------------------------------------------------------
    for json_file in sorted(json_files):
        try:
            print(f"Processing: {os.path.basename(json_file)}")
            with open(json_file, 'r') as f:
                scan_data = json.load(f)

            # Depth Anything → use config yaw;  legacy → use pose yaw
            pose_yaw = get_yaw_from_json(scan_data)
            yaw = pose_yaw if pose_yaw is not None else config_yaw

            mapper.add_scan(scan_data, yaw)
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
            import traceback
            traceback.print_exc()
            continue

    mapper.save()
    return len(json_files)


# ======================================================================
# Main
# ======================================================================

def main():
    """Monitor and process detection files."""

    # ╔══════════════════════════════════════════════════════════════════╗
    # ║  CONFIGURATION — edit these values for your setup              ║
    # ╚══════════════════════════════════════════════════════════════════╝

    mode = "standalone"
    existing_map = os.path.join(BASE_PATH, "room_mapping/office_map.txt")
    existing_json = os.path.join(BASE_PATH, "room_mapping/office.json")
    room_bbox = (23, 10, 40, 24)
    room_name = "MAMAD"

    # ---- Camera wall ----
    #
    #         ┌──── north ────┐
    #         │               │
    #       west            east
    #         │               │
    #         └──── south ────┘
    #
    camera_wall = "west"    # "north" / "south" / "east" / "west"

    # ---- Position along the wall (metres) ----
    #   North/South → distance from WEST wall  (room X)
    #   East/West   → distance from NORTH wall (room Y)
    #
    #   None = middle of the wall (default)
    camera_position_along_wall = None

    # ---- Grid resolution ----
    grid_resolution = 0.05

    # ================================================================

    if room_bbox is not None and existing_map is not None:
        mode = "existing_map"
        print(f"Mode: Existing Map  |  Room: {room_name}  |  bbox: {room_bbox}")
    else:
        print(f"Mode: Standalone  |  Room: {room_name}")

    print(f"Camera wall: {camera_wall}  |  "
          f"Position along wall: {camera_position_along_wall or 'middle'}")
    print(f"Grid resolution: {grid_resolution} m/cell")
    print("\nMonitoring for detection files…  (Ctrl+C to stop)\n")

    bbox_dir = os.path.join(BASE_PATH, "room_mapping/ingest_out")
    last_file_count = 0

    try:
        while True:
            files = glob.glob(os.path.join(bbox_dir, "*.json"))
            if len(files) != last_file_count:
                if files:
                    print(f"\n[{time.strftime('%H:%M:%S')}] {len(files)} file(s) found")
                    n = process_files(
                        mode, existing_map, existing_json, room_bbox, room_name,
                        camera_wall=camera_wall,
                        camera_position_along_wall=camera_position_along_wall,
                        grid_resolution=grid_resolution,
                    )
                    if n > 0:
                        print(f"Processed {n} files → unified_rooms.json + house_map.txt")
                else:
                    print(f"[{time.strftime('%H:%M:%S')}] No detection files")
                last_file_count = len(files)
            time.sleep(2)
    except KeyboardInterrupt:
        print("\n\nStopped by user.")


if __name__ == "__main__":
    main()