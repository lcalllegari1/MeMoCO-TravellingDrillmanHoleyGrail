import json
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

def plot_board_file(json_path, save_path=None, show=False, show_ids=False, marker_size=60):
    with open(json_path, "r") as f:
        data = json.load(f)
        
    holes = data.get("holes", [])
    board = data.get("board", {})
    width = board.get("width", 100)
    height = board.get("height", 100)
    margin = board.get("margin", 0)
    unit = data.get("unit", "mm")
    
    # Styles
    plt.rcParams.update({"font.family": "CMU Sans Serif"})
    fig, ax = plt.subplots(figsize=(6, 6))

    type_style = {
        'mounting_holes': dict(label='Anchors', color='black'),
        'grid':           dict(label='Grid', color='navy'),
        'line':           dict(label='Line', color='teal'),
        'rand':           dict(label='Random', color='orange'),
    }

    # Group holes by type for batch plotting
    holes_by_type = {}
    for h in holes:
        h_type = h.get('type', 'unknown')
        if h_type not in holes_by_type: 
            holes_by_type[h_type] = {'x': [], 'y': []}
        holes_by_type[h_type]['x'].append(h['x'])
        holes_by_type[h_type]['y'].append(h['y'])

    # 1. Draw Scatter Points
    for t in sorted(holes_by_type.keys()):
        coords = holes_by_type[t]
        style = type_style.get(t, dict(label=t.capitalize(), color='gray'))
        
        ax.scatter(
            coords['x'], coords['y'], 
            s=marker_size,  # Slightly larger to ensure text fits comfortably
            c=style['color'], 
            label=style['label'], 
            alpha=0.9, 
            edgecolors='black',
            linewidths=0.5,
            zorder=2
        )

    # 2. Draw IDs (if requested)
    if show_ids:
        for h in holes:
            if 'id' in h:
                ax.text(
                    h['x'], h['y'], 
                    str(h['id']), 
                    color='white',
                    fontsize=6,
                    fontweight='bold',
                    ha='center', 
                    va='center',
                    zorder=3
                )

    # Outlines
    ax.add_patch(patches.Rectangle((0, 0), width, height, linewidth=2, edgecolor='black', facecolor='none', zorder=1))
    if margin > 0:
        ax.add_patch(patches.Rectangle((margin, margin), width-2*margin, height-2*margin, linewidth=1, edgecolor='gray', linestyle='--', facecolor='none', zorder=1))

    # Decoration
    ax.set_title(f"Board: {data.get('id', '?')} ({len(holes)} holes)", fontsize=14)
    ax.set_xlabel(f"X ({unit})")
    ax.set_ylabel(f"Y ({unit})")
    ax.set_aspect('equal')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=4)
    
    # Limits with padding
    ax.set_xlim(-width * 0.05, width * 1.05)
    ax.set_ylim(-height * 0.05, height * 1.05)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved board plot: {save_path}")

    if show:
        plt.show()
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True, help="Input JSON file or Directory")
    parser.add_argument("--output", "-o", help="Output directory (optional)")
    parser.add_argument("--show", action="store_true", help="Display the plot window")
    parser.add_argument("--show-ids", action="store_true", help="Overlay Node IDs inside points")
    
    args = parser.parse_args()
    
    inp = Path(args.input)
    
    if inp.is_file():
        out = Path(args.output) / inp.with_suffix(".png").name if args.output else None
        plot_board_file(inp, out, args.show, args.show_ids)
    elif inp.is_dir():
        out_root = Path(args.output) if args.output else inp
        for f in inp.rglob("*.json"):
            rel = f.relative_to(inp)
            target = out_root / rel.with_suffix(".png")
            target.parent.mkdir(parents=True, exist_ok=True)
            plot_board_file(f, target, args.show, args.show_ids, marker_size=10)

if __name__ == "__main__":
    main()