import yaml
import json
import numpy as np
import argparse
import sys
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

class ClusterGenerator:
    def __init__(self, config_path):
        self.load_config(config_path)

    def load_config(self, config_path):
        with open(config_path, "r") as f:
            specs = yaml.load(f, yaml.SafeLoader)
        # Note: Drill specs aren't strictly needed for K-Means (which is Euclidean), 
        # but kept if we ever want custom distance metrics in the future.
        self.dist_metric = specs.get("dist_metric", "L2")

    def estimate_k(self, coords, n_holes, method="heuristic"):
        """
        Determines K (Number of Clusters).
        """
        if method == "heuristic":
            # Target ~50 nodes per cluster is a good rule of thumb for TSP/Tabu
            target_size = 50
            k = int(n_holes / target_size)
            # Ensure at least 2 clusters, max sqrt(N) to avoid fragmentation
            return max(2, min(k, int(np.sqrt(n_holes))))
            
        # Elbow/Silhouette Method
        k_min = 2
        k_max = min(16, n_holes - 1)
        if n_holes > 200: k_max = 32
        
        best_k = 2
        best_score = -1
        
        # Step=2 to speed up search
        for k in range(k_min, k_max + 1, 2): 
            km = KMeans(n_clusters=k, n_init=5, random_state=42)
            labels = km.fit_predict(coords)
            try:
                score = silhouette_score(coords, labels)
                if score > best_score:
                    best_score = score
                    best_k = k
            except: pass
            
        return best_k

    def process_file(self, input_path, output_path, k_method="elbow", force_k=None):
        try:
            with open(input_path, "r") as f:
                data = json.load(f)
            
            holes = data["holes"]
            n_holes = len(holes)
            
            # Edge case: Too few holes to cluster
            if n_holes < 2:
                print(f"Skipping {input_path.name}: Not enough holes ({n_holes}).")
                return

            # Sort holes by ID to ensure array indices match IDs
            holes_sorted = sorted(holes, key=lambda h: h["id"])
            coords = np.array([[h["x"], h["y"]] for h in holes_sorted])

            # Determine K
            if force_k:
                n_clusters = int(force_k)
            else:
                n_clusters = self.estimate_k(coords, n_holes, k_method)
            
            # Cap K
            n_clusters = min(n_clusters, n_holes)

            # Perform K-Means
            kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
            labels = kmeans.fit_predict(coords)

            # Export
            output_path.parent.mkdir(parents=True, exist_ok=True)
            self.write_dat_file(output_path, n_clusters, labels)
            
        except Exception as e:
            print(f"Error clustering {input_path}: {e}")

    def write_dat_file(self, path, n_clusters, labels):
        """
        Writes the Simple Cluster Format:
        <NUM_MEMBERS> <ID_1> <ID_2> ...
        """
        # Group IDs by label
        grouped_ids = {i: [] for i in range(n_clusters)}
        for original_id, label in enumerate(labels):
            grouped_ids[label].append(original_id)

        with open(path, "w") as f:
            # Iterate strictly in order of cluster ID (0, 1, 2...)
            for i in range(n_clusters):
                ids = grouped_ids[i]
                # row: count id1 id2 ...
                row_values = [len(ids)] + ids
                f.write(" ".join(map(str, row_values)) + "\n")

    def process_directory(self, input_dir, output_dir, k_method, force_k):
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        files = list(input_path.rglob("*.json"))
        print(f"Found {len(files)} boards. Clustering...")
        
        for json_file in files:
            rel_path = json_file.relative_to(input_path)
            target_file = output_path / rel_path.with_suffix(".clusters")
            self.process_file(json_file, target_file, k_method, force_k)
        print("Done.")

def main():
    parser = argparse.ArgumentParser(description="Generate Clusters for Boards.")
    parser.add_argument("--config", "-c", required=True, help="Drill specs YAML")
    parser.add_argument("--input-dir", "-i", required=True)
    parser.add_argument("--output-dir", "-o", required=True)
    parser.add_argument("--method", "-m", default="elbow", choices=["heuristic", "elbow"])
    parser.add_argument("--force-k", "-k", type=int, help="Force specific number of clusters")
    
    args = parser.parse_args()
    
    gen = ClusterGenerator(args.config)
    gen.process_directory(args.input_dir, args.output_dir, args.method, args.force_k)

if __name__ == "__main__":
    main()