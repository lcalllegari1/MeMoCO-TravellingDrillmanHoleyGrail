import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

def load_clusters(dat_path):
    """
    Parses the Simple Cluster Format.
    Returns: dictionary {hole_id: cluster_id}, num_clusters
    """
    with open(dat_path, "r") as f:
        lines = f.readlines()
    
    cluster_map = {} # hole_id -> cluster_id
    cluster_id = 0
    
    for line in lines:
        if not line.strip(): continue
        
        parts = list(map(int, line.strip().split()))
        # Format: count id1 id2 ...
        # parts[0] is count, parts[1:] are ids
        members = parts[1:]
        
        for m in members:
            cluster_map[m] = cluster_id
            
        cluster_id += 1
        
    return cluster_map, cluster_id

def plot_clusters(json_path, clusters_path, save_path=None, show=False, marker_size=60):
    # 1. Load Geometry
    with open(json_path, "r") as f:
        data = json.load(f)
    holes = sorted(data["holes"], key=lambda x: x["id"])
    coords = np.array([[h["x"], h["y"]] for h in holes])
    
    # 2. Load Logic
    cluster_map, n_clusters = load_clusters(clusters_path)
    
    # 3. Assign labels to coordinates
    # Note: If a hole isn't in cluster_map (shouldn't happen), assign -1
    labels = np.array([cluster_map.get(h["id"], -1) for h in holes])
    
    # 4. Re-calculate Centroids for visualization
    centroids = []
    for i in range(n_clusters):
        # Get coords of all points in this cluster
        points = coords[labels == i]
        if len(points) > 0:
            centroids.append(np.mean(points, axis=0))
        else:
            centroids.append([0,0]) # Fallback
    centroids = np.array(centroids)

    # 5. Plotting
    plt.rcParams.update({"font.family": "CMU Sans Serif"})
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Get colormap
    cmap = plt.cm.get_cmap("tab20", n_clusters) if n_clusters <= 20 else plt.cm.get_cmap("nipy_spectral", n_clusters)
    
    # Scatter points
    scatter = ax.scatter(
        coords[:, 0], coords[:, 1], 
        c=labels, cmap=cmap, 
        s=marker_size, alpha=0.8, edgecolors='black', zorder=2, linewidths=0.5
    )
    
    # Scatter Centroids
    if len(centroids) > 0:
        ax.scatter(
            centroids[:, 0], centroids[:, 1],
            s=marker_size*1.5, c='none', marker='o',
            edgecolors='black', linewidths=1.5, zorder=3, label="Centroids"
        )
    
    # Board Outline
    board = data["board"]
    w, h = board["width"], board["height"]
    margin = board.get("margin", 0)
    
    ax.add_patch(patches.Rectangle((0, 0), w, h, linewidth=2, edgecolor='black', facecolor='none', zorder=1))
    if margin > 0:
        ax.add_patch(patches.Rectangle((margin, margin), w-2*margin, h-2*margin, 
                                     linewidth=1, edgecolor='gray', linestyle='--', facecolor='none', zorder=1))
    
    ax.set_title(f"Clustering: {n_clusters} Clusters", fontsize=14, fontweight='bold')
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_aspect('equal')
    
    # Optional Legend if K is small
    if n_clusters <= 10:
        # Create dummy handles for legend
        from matplotlib.lines import Line2D
        handles = []
        for i in range(n_clusters):
            color = cmap(i / (n_clusters - 1)) if n_clusters > 1 else cmap(0)
            handles.append(Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=8, label=f'C{i}'))
        
        ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=min(n_clusters, 5))

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot: {save_path}")
        
    if show:
        plt.show()
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-dir", "-j", required=True, help="Root of JSON boards")
    parser.add_argument("--dat-dir", "-d", required=True, help="Root of Cluster DATs")
    parser.add_argument("--output-dir", "-o", help="Output dir for plots (optional)")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    
    j_root = Path(args.json_dir)
    d_root = Path(args.dat_dir)
    o_root = Path(args.output_dir) if args.output_dir else d_root
    
    # Walk through JSONs and look for matching DATs
    files = list(j_root.rglob("*.json"))
    print(f"Plotting {len(files)} clustering results...")

    for j_file in files:
        rel = j_file.relative_to(j_root)
        d_file = d_root / rel.with_suffix(".clusters")
        
        if d_file.exists():
            target = o_root / rel.with_suffix(".png") # png is faster for batch
            target.parent.mkdir(parents=True, exist_ok=True)
            plot_clusters(j_file, d_file, target, args.show)
        else:
            # Silence this warning usually, unless debugging
            pass

if __name__ == "__main__":
    main()