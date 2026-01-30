import yaml
import json
import numpy as np
import argparse
import sys
from pathlib import Path

# --- Generator Class ---

class MatrixGenerator:
    def __init__(self, config_path):
        self.config_path = Path(config_path)
        self.load_config()

    def load_config(self):
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
            
        with open(self.config_path, "r") as f:
            specs = yaml.load(f, yaml.SafeLoader)
            
        self.dist_metric = specs.get("dist_metric", "L2")
        self.x_vel = float(specs.get("x_vel", 1000.0))
        self.y_vel = float(specs.get("y_vel", 1000.0))
        self.jitter = float(specs.get("jitter", 0.0))
        
        # Validations
        if self.x_vel <= 0 or self.y_vel <= 0:
            raise ValueError("Velocities must be positive.")
            
        print(f"Loaded Drill Specs: Metric={self.dist_metric}, Vx={self.x_vel}, Vy={self.y_vel}")

    def compute_matrix_vectorized(self, holes):
        """
        Computes the NxN cost matrix using NumPy broadcasting.
        This is significantly faster than nested loops for N > 500.
        """
        n = len(holes)
        if n == 0:
            return np.zeros((0, 0))

        # Extract coordinates into (N, 2) array
        # Sorting by ID to ensure matrix index matches hole ID
        holes_sorted = sorted(holes, key=lambda h: h["id"])
        coords = np.array([[h["x"], h["y"]] for h in holes_sorted], dtype=np.float64)

        # 1. Compute Differences (N, N, 2)
        # diffs[i, j, :] = coords[i] - coords[j]
        diffs = np.abs(coords[:, np.newaxis, :] - coords[np.newaxis, :, :])

        # 2. Compute Travel Times
        time_x = diffs[:, :, 0] / self.x_vel
        time_y = diffs[:, :, 1] / self.y_vel

        # 3. Apply Metric
        if self.dist_metric == "L1":
            # Manhattan time (sum of times)
            cost_matrix = time_x + time_y
            
        elif self.dist_metric == "Linf":
            # Chebyshev time (max of times - independent axes)
            cost_matrix = np.maximum(time_x, time_y)
            
        elif self.dist_metric == "L2":
            # Euclidean time (direct path)
            # Note: This assumes diagonal movement combines velocity vectors.
            # Usually strict L2 is sqrt(dx^2 + dy^2) / v_combined.
            # Here we follow your logic: sqrt(tx^2 + ty^2)
            cost_matrix = np.sqrt(time_x**2 + time_y**2)
            
        else:
            raise ValueError(f"Unknown metric: {self.dist_metric}")

        # 4. Apply Jitter (Noise)
        if self.jitter > 0:
            noise = np.random.uniform(0, self.jitter, size=(n, n))
            # Matrix must remain symmetric if directed edges aren't intended
            # For pure noise, symmetry is often broken, but for drilling it usually isn't.
            # Let's keep it simple (asymmetric jitter is fine for testing robustness)
            cost_matrix += noise
            
            # Ensure diagonal is 0? usually yes.
            np.fill_diagonal(cost_matrix, 0.0)

        return cost_matrix

    def process_file(self, input_path, output_path):
        """Reads JSON board, computes matrix, writes .dat file."""
        try:
            with open(input_path, "r") as f:
                data = json.load(f)
            
            n_holes = data["board"]["num_holes"]
            holes = data["holes"]
            
            if len(holes) != n_holes:
                print(f"Warning: Metadata says {n_holes} holes, but found {len(holes)}. Using actual count.")
                n_holes = len(holes)

            # Compute
            matrix = self.compute_matrix_vectorized(holes)
            
            # Save
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Writing manually for specific header format control
            # Format:
            # <N>
            # <row 1>
            # ...
            with open(output_path, "w") as f:
                f.write(f"{n_holes}\n")
                np.savetxt(f, matrix, fmt='%.6f', delimiter=' ')
            
            # print(f"  -> Generated: {output_path.name}") # Optional: keep logs quiet

        except Exception as e:
            print(f"Error processing {input_path}: {e}")

    def process_directory(self, input_root, output_root):
        """Recursively processes all JSON files in input_root."""
        input_path = Path(input_root)
        output_path = Path(output_root)
        
        if not input_path.exists():
            print(f"Error: Input directory {input_path} does not exist.")
            return

        print(f"Scanning {input_path}...")
        files = list(input_path.rglob("*.json"))
        print(f"Found {len(files)} board files.")

        for json_file in files:
            # Calculate relative path to maintain folder structure
            # e.g., data/boards/benchmark/set_a/board.json -> benchmark/set_a/board.json
            rel_path = json_file.relative_to(input_path)
            
            # Target .dat file
            target_file = output_path / rel_path.with_suffix(".dat")
            
            self.process_file(json_file, target_file)
        
        print("Done.")


# --- Main Entry Point ---

def main():
    parser = argparse.ArgumentParser(description="Generate Cost Matrices from Board JSONs.")
    
    parser.add_argument(
        "--config", "-c", 
        type=str, 
        required=True, 
        help="Path to the Drill Specs YAML configuration file."
    )
    
    parser.add_argument(
        "--input-dir", "-i", 
        type=str, 
        required=True, 
        help="Root directory containing Board JSON files (e.g., data/boards)."
    )
    
    parser.add_argument(
        "--output-dir", "-o", 
        type=str, 
        required=True, 
        help="Root directory for output .dat files (e.g., data/instances)."
    )
    
    args = parser.parse_args()

    generator = MatrixGenerator(args.config)
    generator.process_directory(args.input_dir, args.output_dir)

if __name__ == "__main__":
    main()