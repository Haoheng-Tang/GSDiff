"""
Randomly sample 5 floor plans from the MSD dataset and render each as:
  Left panel  — Floor plan with actual room polygons, colored by room type
  Right panel — Adjacency graph with nodes at spatial centroid positions,
                colored by room type, edges colored by connectivity type

Output: test_outputs/msd_floorplan_plots/floorplan_{floor_id}.png
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection
import networkx as nx

from datasets.msd_topology import (
    list_floor_ids, load_floor_plan,
    ROOM_TYPE_NAMES, ROOM_TYPE_SHORT, ROOM_TYPE_COLORS, EDGE_COLORS,
)

# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------
SPLIT = 'train'
N_PLANS = 5
RANDOM_SEED = 42
OUT_DIR = Path('test_outputs/msd_floorplan_plots')
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def get_bbox(G):
    """Return (xmin, xmax, ymin, ymax) bounding box over all polygon vertices."""
    xs, ys = [], []
    for _, d in G.nodes(data=True):
        for x, y in d['geometry']:
            xs.append(x)
            ys.append(y)
    return min(xs), max(xs), min(ys), max(ys)


def plot_floor_plan(ax, G):
    """Draw room polygons on ax, colored by room type."""
    xmin, xmax, ymin, ymax = get_bbox(G)
    span = max(xmax - xmin, ymax - ymin, 1e-3)

    ax.set_xlim(xmin - span * 0.05, xmax + span * 0.05)
    ax.set_ylim(ymin - span * 0.05, ymax + span * 0.05)
    ax.set_aspect('equal')
    ax.set_facecolor('#F8F8F8')
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    for nid, d in G.nodes(data=True):
        verts = np.array(d['geometry'])
        rt = d['room_type']
        color = ROOM_TYPE_COLORS.get(rt, (0.8, 0.8, 0.8))

        patch = MplPolygon(verts, closed=True,
                           facecolor=color, edgecolor='#444444',
                           linewidth=0.8, alpha=0.88, zorder=2)
        ax.add_patch(patch)

        # centroid label
        cx, cy = float(d['centroid'][0]), float(d['centroid'][1])
        label = ROOM_TYPE_SHORT.get(rt, str(rt))
        ax.text(cx, cy, label, ha='center', va='center',
                fontsize=5.5, fontweight='bold', color='#222222', zorder=4)

    ax.set_title('Floor Plan', fontsize=9, pad=4)


def plot_graph(ax, G):
    """Draw the adjacency graph with nodes at spatial centroid positions."""
    xmin, xmax, ymin, ymax = get_bbox(G)
    span = max(xmax - xmin, ymax - ymin, 1e-3)

    ax.set_xlim(xmin - span * 0.1, xmax + span * 0.1)
    ax.set_ylim(ymin - span * 0.1, ymax + span * 0.1)
    ax.set_aspect('equal')
    ax.set_facecolor('#FAFAFA')
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # build pos dict
    pos = {}
    node_colors = []
    for nid, d in G.nodes(data=True):
        cx, cy = float(d['centroid'][0]), float(d['centroid'][1])
        pos[nid] = (cx, cy)
        rt = d['room_type']
        node_colors.append(ROOM_TYPE_COLORS.get(rt, (0.8, 0.8, 0.8)))

    # draw edges grouped by connectivity type
    for ctype, color in EDGE_COLORS.items():
        edge_list = [(u, v) for u, v, d in G.edges(data=True)
                     if d.get('connectivity') == ctype]
        if edge_list:
            nx.draw_networkx_edges(G, pos, edgelist=edge_list,
                                   edge_color=color, width=1.4,
                                   alpha=0.75, ax=ax)

    # draw nodes
    nx.draw_networkx_nodes(G, pos,
                           node_color=node_colors,
                           node_size=220,
                           edgecolors='#333333',
                           linewidths=0.8,
                           ax=ax)

    # draw short labels
    labels = {nid: ROOM_TYPE_SHORT.get(d['room_type'], str(d['room_type']))
              for nid, d in G.nodes(data=True)}
    nx.draw_networkx_labels(G, pos, labels=labels,
                            font_size=4.5, font_weight='bold',
                            font_color='#111111', ax=ax)

    ax.set_title('Adjacency Graph', fontsize=9, pad=4)


def make_legend_handles():
    """Return patch + line handles for a shared legend."""
    handles = []
    for rt, name in ROOM_TYPE_NAMES.items():
        color = ROOM_TYPE_COLORS.get(rt, (0.8, 0.8, 0.8))
        handles.append(mpatches.Patch(facecolor=color, edgecolor='#444444',
                                      linewidth=0.5, label=name))
    handles.append(mlines.Line2D([], [], color=EDGE_COLORS['door'],
                                 linewidth=1.5, label='Door'))
    handles.append(mlines.Line2D([], [], color=EDGE_COLORS['entrance'],
                                 linewidth=1.5, label='Entrance'))
    handles.append(mlines.Line2D([], [], color=EDGE_COLORS['passage'],
                                 linewidth=1.5, label='Passage'))
    return handles


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    random.seed(RANDOM_SEED)
    all_ids = list_floor_ids(SPLIT)
    chosen = random.sample(all_ids, N_PLANS)
    print(f'Selected floor IDs: {chosen}')

    legend_handles = make_legend_handles()

    for fid in chosen:
        G = load_floor_plan(fid, SPLIT)
        n_rooms = G.number_of_nodes()
        n_edges = G.number_of_edges()

        print(f'\nFloor {fid}: {n_rooms} rooms, {n_edges} edges')
        rt_counts = {}
        for _, d in G.nodes(data=True):
            name = ROOM_TYPE_NAMES.get(d['room_type'], f"type_{d['room_type']}")
            rt_counts[name] = rt_counts.get(name, 0) + 1
        for name, cnt in sorted(rt_counts.items(), key=lambda x: -x[1]):
            print(f'  {name}: {cnt}')

        fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
        fig.suptitle(
            f'MSD Floor Plan  |  floor_id={fid}  |  {n_rooms} rooms, {n_edges} connections',
            fontsize=10, fontweight='bold', y=1.01
        )

        plot_floor_plan(axes[0], G)
        plot_graph(axes[1], G)

        # shared legend below both panels
        fig.legend(handles=legend_handles,
                   loc='lower center',
                   ncol=6,
                   fontsize=7.5,
                   frameon=True,
                   bbox_to_anchor=(0.5, -0.07),
                   borderaxespad=0)

        plt.tight_layout(pad=1.5)
        out_path = OUT_DIR / f'floorplan_{fid}.png'
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'  Saved → {out_path}')

    # also save a combined 5-up overview figure (floor plans only)
    fig_all, axes_all = plt.subplots(2, N_PLANS, figsize=(4 * N_PLANS, 9))
    for col, fid in enumerate(chosen):
        G = load_floor_plan(fid, SPLIT)
        plot_floor_plan(axes_all[0, col], G)
        axes_all[0, col].set_title(f'id={fid}  ({G.number_of_nodes()} rooms)', fontsize=8)
        plot_graph(axes_all[1, col], G)
        axes_all[1, col].set_title('')

    fig_all.text(0.01, 0.73, 'Floor Plan', va='center', rotation='vertical',
                 fontsize=9, fontweight='bold')
    fig_all.text(0.01, 0.27, 'Graph', va='center', rotation='vertical',
                 fontsize=9, fontweight='bold')
    fig_all.legend(handles=legend_handles,
                   loc='lower center', ncol=6, fontsize=7,
                   frameon=True, bbox_to_anchor=(0.5, -0.04))
    plt.tight_layout(pad=1.2)
    overview_path = OUT_DIR / 'overview_5plans.png'
    fig_all.savefig(overview_path, dpi=150, bbox_inches='tight')
    plt.close(fig_all)
    print(f'\nOverview saved → {overview_path}')


if __name__ == '__main__':
    main()
