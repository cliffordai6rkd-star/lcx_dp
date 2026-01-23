#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def collect_points(data_dir):
    # Load all inference JSON files and bucket points by ee_id + success flag.
    points_by_ee, samples = {}, []
    for path in sorted(data_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict):
            continue
        for _, ee_map in data.items():
            if not isinstance(ee_map, dict):
                continue
            for ee_id, items in ee_map.items():
                if not isinstance(items, list):
                    continue
                bucket = points_by_ee.setdefault(str(ee_id), {"success": [], "fail": []})
                for entry in items:
                    if not isinstance(entry, (list, tuple)) or len(entry) < 2:
                        continue
                    pose, ok = entry[0], entry[1]
                    if not isinstance(pose, (list, tuple)) or len(pose) < 3:
                        continue
                    x, y, z = pose[:3]
                    if ok:
                        bucket["success"].append((x, y, z)); samples.append((x, y, z, True))
                    else:
                        bucket["fail"].append((x, y, z)); samples.append((x, y, z, False))
    return points_by_ee, samples


def set_equal_aspect(ax, points):
    # Keep x/y/z scales equal to avoid visual distortion.
    if not points:
        return
    xs, ys, zs = zip(*points)
    x_min, x_max = min(xs), max(xs); y_min, y_max = min(ys), max(ys); z_min, z_max = min(zs), max(zs)
    max_range = max(x_max - x_min, y_max - y_min, z_max - z_min)
    if max_range <= 0:
        return
    mid_x = (x_max + x_min) / 2.0; mid_y = (y_max + y_min) / 2.0; mid_z = (z_max + z_min) / 2.0
    half = max_range / 2.0
    ax.set_xlim(mid_x - half, mid_x + half); ax.set_ylim(mid_y - half, mid_y + half); ax.set_zlim(mid_z - half, mid_z + half)


def auto_cluster_radius(samples, max_points=1000):
    # Heuristic radius: half of median nearest-neighbor distance.
    coords = [(x, y, z) for x, y, z, _ in samples[:max_points]]
    if len(coords) < 2:
        return None
    nearest = []
    for i, (x1, y1, z1) in enumerate(coords):
        best = None
        for j, (x2, y2, z2) in enumerate(coords):
            if i == j:
                continue
            dx = x1 - x2; dy = y1 - y2; dz = z1 - z2
            d = math.sqrt(dx * dx + dy * dy + dz * dz)
            if d > 0 and (best is None or d < best):
                best = d
        if best is not None:
            nearest.append(best)
    if not nearest:
        return None
    nearest.sort()
    return max(nearest[len(nearest) // 2] * 0.5, 1e-6)


def cluster_stats(samples, radius):
    # Single-linkage clustering by radius using union-find.
    if radius <= 0 or not samples:
        return None
    n = len(samples); parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i

    r2 = radius * radius
    for i in range(n):
        x1, y1, z1, _ = samples[i]
        for j in range(i + 1, n):
            x2, y2, z2, _ = samples[j]
            dx = x1 - x2; dy = y1 - y2; dz = z1 - z2
            if dx * dx + dy * dy + dz * dz <= r2:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[rj] = ri

    clusters = {}
    for idx in range(n):
        clusters.setdefault(find(idx), []).append(idx)

    stats = []
    for cid, idxs in enumerate(clusters.values()):
        xs = [samples[i][0] for i in idxs]; ys = [samples[i][1] for i in idxs]; zs = [samples[i][2] for i in idxs]
        total = len(idxs); success = sum(1 for i in idxs if samples[i][3])
        center = (sum(xs) / total, sum(ys) / total, sum(zs) / total)
        stats.append({
            "id": cid,
            "min_bound": [min(xs), min(ys), min(zs)],
            "max_bound": [max(xs), max(ys), max(zs)],
            "center": [center[0], center[1], center[2]],
            "success": success,
            "total": total,
            "success_rate": success / total if total else 0.0,
        })

    return {"cluster_radius": radius, "total_points": len(samples), "cluster_count": len(stats), "clusters": stats}


def emit_stats(stats, output_path):
    # Save cluster stats to JSON, or print if no path is given.
    if stats is None:
        return
    text = json.dumps(stats, indent=2, sort_keys=True)
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")
    else:
        print(text)


def box_faces(min_bound, max_bound, padding=0.0):
    # Build 6 quad faces for an axis-aligned box.
    x0, y0, z0 = min_bound; x1, y1, z1 = max_bound
    if padding > 0:
        if x0 == x1: x0 -= padding; x1 += padding
        if y0 == y1: y0 -= padding; y1 += padding
        if z0 == z1: z0 -= padding; z1 += padding
    v000 = (x0, y0, z0); v001 = (x0, y0, z1); v010 = (x0, y1, z0); v011 = (x0, y1, z1)
    v100 = (x1, y0, z0); v101 = (x1, y0, z1); v110 = (x1, y1, z0); v111 = (x1, y1, z1)
    return [[v000, v100, v110, v010], [v001, v101, v111, v011], [v000, v100, v101, v001],
            [v010, v110, v111, v011], [v000, v010, v011, v001], [v100, v110, v111, v101]]


def visualize(data_dir, output_path, cluster_radius, stats_path):
    # Scatter plot: ee_id color, success checkmark, fail x.
    points_by_ee, samples = collect_points(data_dir)
    if not samples:
        raise SystemExit(f"No valid points found under: {data_dir}")

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    cmap = plt.get_cmap("tab20")

    def ee_key(v):
        try:
            return (0, int(v))
        except ValueError:
            return (1, v)

    any_success = False; any_fail = False
    for idx, ee_id in enumerate(sorted(points_by_ee, key=ee_key)):
        color = cmap(idx % cmap.N)
        success_points = points_by_ee[ee_id]["success"]; fail_points = points_by_ee[ee_id]["fail"]
        if success_points:
            any_success = True; xs, ys, zs = zip(*success_points)
            ax.scatter(xs, ys, zs, marker="$\\checkmark$", s=80, c=[color], alpha=0.85, label=f"ee {ee_id}")
        if fail_points:
            any_fail = True; xs, ys, zs = zip(*fail_points)
            ax.scatter(xs, ys, zs, marker="x", s=50, c=[color], alpha=0.85, label=None if success_points else f"ee {ee_id}")

    # Heatmap: cluster bounds colored by success rate.
    radius = cluster_radius
    if radius is None:
        radius = auto_cluster_radius(samples) or 0.01
    stats = cluster_stats(samples, radius) if radius > 0 else None
    emit_stats(stats, stats_path)

    if stats:
        norm = Normalize(vmin=0.0, vmax=1.0); heat_cmap = plt.get_cmap("viridis")
        for cluster in stats["clusters"]:
            color = heat_cmap(norm(cluster["success_rate"]))
            faces = box_faces(cluster["min_bound"], cluster["max_bound"], padding=radius * 0.25)
            box = Poly3DCollection(faces, facecolors=[color], edgecolors=[color], linewidths=0.6, alpha=0.25)
            ax.add_collection3d(box)
        sm = ScalarMappable(norm=norm, cmap=heat_cmap); sm.set_array([])
        fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.08, label="cluster success_rate")

    all_points = [p[:3] for p in samples]
    if stats:
        for cluster in stats["clusters"]:
            all_points.append(tuple(cluster["min_bound"])); all_points.append(tuple(cluster["max_bound"]))
    set_equal_aspect(ax, all_points)

    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_title(f"EE pose success map ({len(samples)} points)")

    handles, labels = ax.get_legend_handles_labels(); marker_handles = []
    if any_success:
        marker_handles.append(Line2D([0], [0], marker="$\\checkmark$", color="k", linestyle="None", markersize=10, label="success"))
    if any_fail:
        marker_handles.append(Line2D([0], [0], marker="x", color="k", linestyle="None", markersize=8, label="fail"))
    if handles or marker_handles:
        ax.legend(handles + marker_handles, labels + [h.get_label() for h in marker_handles])

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize EE pose success/failure points in 3D space.")
    parser.add_argument("--data-dir", type=Path, default=Path("dataset/data/infer_bp_umi_fr3"), help="Directory containing inference JSON files.")
    parser.add_argument("--output", type=Path, default=None, help="Optional output image path (png). If omitted, show the plot window.")
    parser.add_argument("--cluster-radius", type=float, default=None, help="Clustering radius (meters). If omitted, auto-estimate; set <=0 to disable heatmap.")
    parser.add_argument("--heatmap-json", type=Path, default=None, help="Optional output path for cluster stats JSON. If omitted, print to terminal.")
    args = parser.parse_args()
    if not args.data_dir.exists():
        raise SystemExit(f"Data directory not found: {args.data_dir}")
    visualize(args.data_dir, args.output, args.cluster_radius, args.heatmap_json)

if __name__ == "__main__":
    main()
