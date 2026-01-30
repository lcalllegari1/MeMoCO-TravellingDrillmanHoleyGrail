import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from pathlib import Path
import sys

def load_tour_file(path):
    """
    Reads a tour sequence from a text file.
    Supports space, comma, or newline separation.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Tour file not found: {path}")
    
    with open(path, "r") as f:
        content = f.read()
        # Replace commas with spaces to handle CSV-like formats
        content = content.replace(",", " ")
        parts = content.split()
        
    try:
        return [int(x) for x in parts]
    except ValueError:
        print("Error: Tour file must contain only integers.")
        sys.exit(1)

def plot_tour(
    board_path,
    tour,
    save_path=None,
    show=False,
    # --- Aesthetics ---
    figsize=(6, 6),
    marker_size=60,
    title_text=None,
    font_family="CMU Sans Serif", # Use 'serif' for LaTeX-like look
    # --- Tour Style ---
    tour_width=1.4,
    tour_alpha=0.8,
    tour_cmap="viridis", # 'turbo' or 'plasma' are also great
    show_arrows=True,
    arrow_interval=1,    # Plot arrow every N segments (prevent clutter)
    arrow_size=8,
    # --- Start Node ---
    highlight_start=True,
    start_color="red",
    start_ring_scale=2.25
):
    # 1. Load Board Data
    board_path = Path(board_path)
    with open(board_path, "r") as f:
        data = json.load(f)

    holes = data.get("holes", [])
    board = data.get("board", {})
    width = board.get("width", 100)
    height = board.get("height", 100)
    margin = board.get("margin", 0)
    unit = data.get("unit", "mm")

    # Map ID -> Coordinates
    id_to_coords = {h['id']: (h['x'], h['y']) for h in holes}

    # Setup Plot
    plt.rcParams.update({"font.family": font_family})
    fig, ax = plt.subplots(figsize=figsize)

    # 2. Plot Tour (Gradient Line Collection)
    tour_proxy = None
    start_proxy = None
    
    if tour and len(tour) > 1:
        # Validate Tour
        coords = []
        for nid in tour:
            if nid in id_to_coords:
                coords.append(id_to_coords[nid])
            else:
                # Optional: Warn if node missing (e.g. subset tour)
                pass
        
        points = np.array(coords)
        if len(points) > 1:
            # Create segments: (x1, y1) -> (x2, y2)
            # Shape: (N-1, 2, 2)
            segments = np.concatenate([
                points[:-1].reshape(-1, 1, 2), 
                points[1:].reshape(-1, 1, 2)
            ], axis=1)

            # Color Map based on index
            norm = mcolors.Normalize(vmin=0, vmax=len(segments))
            cmap = cm.get_cmap(tour_cmap)
            colors = cmap(norm(np.arange(len(segments))))

            # A. Draw Lines
            lc = LineCollection(segments, colors=colors, linewidths=tour_width, alpha=tour_alpha, zorder=2)
            ax.add_collection(lc)

            # B. Draw Arrows
            if show_arrows:
                # Calculate midpoints and angles
                mid_x = (points[:-1, 0] + points[1:, 0]) / 2
                mid_y = (points[:-1, 1] + points[1:, 1]) / 2
                dx = points[1:, 0] - points[:-1, 0]
                dy = points[1:, 1] - points[:-1, 1]
                angles = np.degrees(np.arctan2(dy, dx))

                # Slice to avoid clutter
                indices = range(0, len(segments), arrow_interval)
                
                for i in indices:
                    # Marker rotation logic
                    # (3, 0, angle) creates a triangle rotated by angle
                    # -90 adjustment depends on the triangle marker orientation default
                    marker_style = (3, 0, angles[i] - 90)
                    
                    ax.scatter(
                        mid_x[i], mid_y[i], 
                        marker=marker_style, s=arrow_size**2, 
                        color=colors[i], edgecolors='none', 
                        alpha=tour_alpha, zorder=2.1
                    )

            # Legend Entry
            tour_proxy = Line2D([0], [0], color=cmap(0.5), linewidth=tour_width, label='Tour Path')

            # C. Highlight Start
            if highlight_start:
                sx, sy = points[0]
                # Determine hole size from the scatter plot below (approx)
                # We assume a default size for scaling
                base_s = 60 
                
                ax.scatter(
                    sx, sy, 
                    s=base_s * start_ring_scale, 
                    facecolors='none', edgecolors=start_color, linewidth=2, 
                    zorder=4, label='Start Node'
                )
                start_proxy = Line2D([0], [0], marker='o', color='w', label='Start Node',
                                     markerfacecolor='none', markeredgecolor=start_color, 
                                     markeredgewidth=2, markersize=10)

    # 3. Plot Holes (Context)
    type_style = {
        'mounting_holes': dict(label='Anchors', color='black'),
        'grid':           dict(label='Grid', color='navy'),
        'line':           dict(label='Line', color='teal'),
        'rand':           dict(label='Random', color='orange'),
    }
    
    holes_by_type = {}
    for h in holes:
        t = h.get('type', 'unknown')
        if t not in holes_by_type: holes_by_type[t] = {'x': [], 'y': []}
        holes_by_type[t]['x'].append(h['x'])
        holes_by_type[t]['y'].append(h['y'])

    for t in sorted(holes_by_type.keys()):
        c = holes_by_type[t]
        style = type_style.get(t, dict(label=t.capitalize(), color='gray'))
        ax.scatter(c['x'], c['y'], s=marker_size, c=style['color'], label=style['label'], alpha=0.9, edgecolors='black', zorder=1, linewidths=0.5)

    # 4. Board Outline
    ax.add_patch(patches.Rectangle((0, 0), width, height, linewidth=2, edgecolor='black', facecolor='none', zorder=0))
    if margin > 0:
        ax.add_patch(patches.Rectangle((margin, margin), width-2*margin, height-2*margin, 
                                     linewidth=1, edgecolor='gray', linestyle='--', facecolor='none', zorder=0))

    # 5. Decoration
    if title_text is None:
        cost_str = ""
        if tour:
            cost_str = f" (Tour Length: {len(tour)})"
            cost_str = f""
        title_text = f"Optimized Tour: instance {data.get('id', '?')}{cost_str}"

    ax.set_title(title_text, fontsize=14, fontweight='bold')
    ax.set_xlabel(f"X ({unit})")
    ax.set_ylabel(f"Y ({unit})")
    ax.set_aspect('equal')
    
    # Custom Legend Order
    handles, labels = ax.get_legend_handles_labels()
    final_h, final_l = [], []
    
    # Priority Items
    if start_proxy:
        final_h.append(start_proxy)
        final_l.append("Start Node")
    if tour_proxy:
        final_h.append(tour_proxy)
        final_l.append("Tour Path")
        
    # Append the rest (avoiding duplicates)
    for h, l in zip(handles, labels):
        if l not in final_l and l != "Start Node":
            final_h.append(h)
            final_l.append(l)
            
    ax.legend(handles=final_h, labels=final_l, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3)
    
    # Pad limits
    ax.set_xlim(-width * 0.05, width * 1.05)
    ax.set_ylim(-height * 0.05, height * 1.05)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved tour plot: {save_path}")

    if show:
        plt.show()
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description="Visualize a TSP Tour on a PCB Board.")
    parser.add_argument("--board", "-b", required=True, help="Path to Board JSON file")
    parser.add_argument("--tour", "-t", required=True, help="Path to Tour file (space/newline separated integers)")
    parser.add_argument("--output", "-o", help="Path to save PDF/PNG")
    parser.add_argument("--show", action="store_true", help="Open window")
    parser.add_argument("--arrows", action="store_true", default=True, help="Show direction arrows")
    parser.add_argument("--no-arrows", action="store_false", dest="arrows")
    parser.add_argument("--arrow-interval", type=int, default=5, help="Draw arrow every N segments")
    
    args = parser.parse_args()

    # Load Tour
    tour = load_tour_file(args.tour)
    
    # Plot
    plot_tour(
        args.board, 
        tour, 
        save_path=args.output, 
        show=args.show,
        show_arrows=args.arrows,
        arrow_interval=args.arrow_interval
    )

if __name__ == "__main__":
    main()