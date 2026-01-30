import yaml
import json
import numpy as np
import argparse
import sys
from pathlib import Path

# --- Utility Functions ---

def build_filename(n_holes, unique_id, ext="json"):
    return f"n{n_holes}_{unique_id}.{ext}"

# --- Generator Class ---

class BoardGenerator: 
    def __init__(self, output_dir_override=None):
        self.output_dir_override = Path(output_dir_override) if output_dir_override else None
        
        # State variables
        self.holes = []
        self.hole_counter = 0
        self.width = 0
        self.height = 0
        self.margin = 0
        self.min_spacing = 0
        self.anchors = {}
        self.components = []
        self.n_instances = 0
        self.board_unit = "mm"
        self.target_dir = None
        # NEW: Track bounding boxes of placed components to prevent overlap
        self.placed_bboxes = [] 

    def process_config(self, config):
        """
        Parses a single configuration dictionary and generates instances.
        """
        self._parse(config)
        
        print(f"Processing config idx={config.get('idx', '?')} -> Generating {self.n_instances} instances...")
        
        for i in range(self.n_instances):
            self.generate_instance()
            self.save_instance(config.get("idx"))
            
    def _parse(self, config):
        self.width = config["board"]["width"]
        self.height = config["board"]["height"]
        self.margin = config["board"]["margin"]
        self.min_spacing = config["board"]["min_spacing"]
        self.anchors = config["board"]["mounting_holes"]
        self.components = config["components"]
        
        self.n_instances = config["metadata"]["n_instances"]
        self.board_unit = config["metadata"]["unit"]
        
        # Determine output directory
        if self.output_dir_override:
            self.target_dir = self.output_dir_override
            if "target_dir" in config["metadata"]:
                 self.target_dir = self.target_dir / config["metadata"]["target_dir"]
        else:
            base = Path(config["metadata"].get("base_dir_path", "data/boards/"))
            target = Path(config["metadata"].get("target_dir", "generated/"))
            self.target_dir = base / target

    def add_hole(self, x, y, hole_type):
        self.holes.append({
            "id": self.hole_counter,
            "type": hole_type,
            "x": float(x),
            "y": float(y)
        })
        self.hole_counter += 1

    def generate_instance(self):
        # Reset state for new instance
        self.holes = []
        self.hole_counter = 0
        self.placed_bboxes = [] 

        if self.anchors.get("enabled", False):
            self.generate_anchors(self.anchors)
        
        for comp in self.components:
            self.generate_component(comp["type"], comp["params"])
            
    def generate_anchors(self, anchors):
        w, h, m = self.width, self.height, self.margin
        # Anchors don't need bbox checking, they are placed first absolutely
        if anchors.get("corners"):
            self.generate_corner_holes(w, h, m)
        if anchors.get("center"):
            self.generate_center_hole(w, h)
        if anchors.get("edges"):
            self.generate_edge_holes(w, h, m)
            
    def generate_corner_holes(self, w, h, m):
        corners = [
            (m, m), (w - m, m), (m, h - m), (w - m, h - m)
        ]
        for x, y in corners:
            self.add_hole(x, y, hole_type="mounting_holes")

    def generate_center_hole(self, w, h):
        self.add_hole(w / 2, h / 2, hole_type="mounting_holes")

    def generate_edge_holes(self, w, h, m):
        edges = [
            (m, h / 2), (w - m, h / 2), (w / 2, m), (w / 2, h - m)
        ]
        for x, y in edges:
            self.add_hole(x, y, hole_type="mounting_holes")
            
    def generate_component(self, comp_type, params):
        repeat = params.get("repeat", 1)
        for _ in range(repeat):
            # Special case for random holes (logic inside generation)
            if comp_type == "rand":
                self.generate_random_holes(params["n_holes"], self.min_spacing)
                continue
            
            # Geometry generation
            holes = []
            if comp_type == "grid":
                holes = self.generate_grid(
                    params["rows"], params["cols"], 
                    params["row_spacing"], params["col_spacing"]
                )
            elif comp_type == "line":
                holes = self.generate_line(
                    params["n_holes"], params["spacing"]
                )
            
            # Placement
            if len(holes) > 0:
                self.place_component(holes, comp_type, params.get("angle_deg", None))
    
    def generate_random_holes(self, n, spacing, max_attempts=1000):
        w, h, m = self.width, self.height, self.margin
        local_holes = []
        # Random holes don't have a bounding box footprint, they just check hole spacing
        for _ in range(n):
            for _ in range(max_attempts):
                x = np.random.uniform(low=m, high=w - m)
                y = np.random.uniform(low=m, high=h - m)
                if self.feasible([(x, y)], spacing):
                    self.add_hole(x, y, hole_type="rand")
                    local_holes.append([x, y])
                    break
        return np.array(local_holes)
    
    def generate_grid(self, rows, cols, row_spacing, col_spacing):
        holes = []
        h_total = row_spacing * (rows - 1)
        w_total = col_spacing * (cols - 1)
        for r in range(rows):
            for c in range(cols):
                x = c * col_spacing - w_total / 2
                y = r * row_spacing - h_total / 2 
                holes.append((x, y))
        return np.array(holes)
    
    def generate_line(self, n_holes, spacing):
        holes = []
        w_total = spacing * (n_holes - 1)
        for i in range(n_holes):
            holes.append(((i * spacing) - w_total / 2, 0))
        return np.array(holes)
    
    def feasible(self, candidate_holes, spacing):
        # Optimization: In a real scenario with 3000 holes, a KD-Tree is better.
        # But for generation speed < 1s, brute force loop is acceptable.
        for hx, hy in candidate_holes:
            for h in self.holes:
                if np.hypot(hx - h["x"], hy - h["y"]) < spacing: 
                    return False
        return True

    def bounding_box(self, holes):
        # Returns (min_x, min_y, max_x, max_y)
        return np.min(holes, axis=0)[0], np.min(holes, axis=0)[1], \
               np.max(holes, axis=0)[0], np.max(holes, axis=0)[1]

    def bbox_intersects(self, boxA, boxB):
        # NEW: Check if two AABBs intersect with a small buffer for visual clearance
        # box = (minx, miny, maxx, maxy)
        buffer = self.min_spacing / 2.0

        # If one rectangle is to the left of the other
        if (boxA[0] - buffer > boxB[2] + buffer) or (boxB[0] - buffer > boxA[2] + buffer):
            return False
        # If one rectangle is above the other
        if (boxA[3] + buffer < boxB[1] - buffer) or (boxB[3] + buffer < boxA[1] - buffer):
            return False
        return True
    
    def attempt_place(self, holes, width, height, margin, spacing):
        min_x, min_y, max_x, max_y = self.bounding_box(holes)
        comp_w, comp_h = max_x - min_x, max_y - min_y
        
        valid_w = width - 2 * margin
        valid_h = height - 2 * margin
        
        if comp_w > valid_w or comp_h > valid_h:
            return None
        
        x_min_offset = margin - min_x
        x_max_offset = width - margin - max_x
        y_min_offset = margin - min_y
        y_max_offset = height - margin - max_y

        offset_x = np.random.uniform(x_min_offset, x_max_offset)
        offset_y = np.random.uniform(y_min_offset, y_max_offset)
        
        candidate_holes = holes + [offset_x, offset_y]
        
        # NEW: Calculate the final bounding box of the candidate
        cand_bbox = (min_x + offset_x, min_y + offset_y, max_x + offset_x, max_y + offset_y)

        # NEW: Check against existing component bounding boxes
        for existing_bbox in self.placed_bboxes:
            if self.bbox_intersects(cand_bbox, existing_bbox):
                return None # Overlap detected

        # Existing check against individual holes
        if self.feasible(candidate_holes, spacing):
            # Return both holes and the bbox
            return candidate_holes, cand_bbox
        return None
    
    def place_component(self, holes, comp_type, angle=None, max_attempts=2000):
        # Increased max_attempts as placement is now harder
        for _ in range(max_attempts):
            rotated = self.rotate_holes(holes, angle)
            result = self.attempt_place(
                rotated, self.width, self.height, self.margin, self.min_spacing
            )
            if result is not None:
                candidate_holes, cand_bbox = result
                for x, y in candidate_holes:
                    self.add_hole(x, y, comp_type)
                # NEW: Save the bounding box footprint
                self.placed_bboxes.append(cand_bbox)
                return
    
    def rotate_holes(self, holes, angle):
        if angle == 0:
            return holes
        
        if angle is None: 
            rad = np.random.uniform(0, 2 * np.pi)
        else:
            rad = np.deg2rad(angle)
            
        cos, sin = np.cos(rad), np.sin(rad)
        rmat = np.array([[cos, -sin], [sin, cos]]) 
        return holes @ rmat.T
        
    def save_instance(self, config_id):
        # Generate a short random ID for filename uniqueness
        rand_id = f"{np.random.randint(0, 0xFFFFFFFF):08x}"
        
        # Use config ID in filename if available for easier identification
        filename = f"n{self.hole_counter}_{rand_id}.json"
        
        final_dir = self.target_dir
        final_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = final_dir / filename
        
        data = {
            "id": f"{config_id}_{rand_id}",
            "unit": self.board_unit,
            "board": {
                "width": self.width,
                "height": self.height,
                "margin": self.margin,
                "num_holes": self.hole_counter,
            },
            "holes": self.holes
        }
        
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)
        # print(f"  -> Saved: {file_path}")

# --- Main Entry Point (for standalone usage) ---

def main():
    parser = argparse.ArgumentParser(description="Generate PCB Board Instances from YAML config.")
    parser.add_argument("--config", "-c", type=str, required=True, help="Path to YAML config.")
    parser.add_argument("--indices", "-i", type=int, nargs="+", help="Specific indices to run.")
    parser.add_argument("--output-dir", "-o", type=str, help="Override output directory.")
    parser.add_argument("--seed", "-s", type=int, help="Set global random seed.")

    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found at {config_path}")
        sys.exit(1)

    with open(config_path, "r") as f:
        all_configs = list(yaml.load_all(f, yaml.SafeLoader))

    if args.indices:
        selected_configs = [c for c in all_configs if c.get("idx") in args.indices]
    else:
        selected_configs = all_configs

    generator = BoardGenerator(output_dir_override=args.output_dir)
    for config in selected_configs:
        generator.process_config(config)

if __name__ == "__main__":
    main()