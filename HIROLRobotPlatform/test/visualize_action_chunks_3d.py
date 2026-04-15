#!/usr/bin/env python3
"""Visualize inference action chunks in one 3D space.

Each chunk is rendered with a unique color (mapped by chunk index) so points from
the same chunk are visually grouped. Optionally draws dashed links from chunk end
to the next chunk start to expose discontinuities.
"""

import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from scipy.spatial.transform import Rotation as R


def _set_equal_axes(ax, points: np.ndarray) -> None:
    if points.size == 0:
        return
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = np.max(maxs - mins) / 2.0
    if radius <= 0:
        radius = 1e-3
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def _load_init_pose(data_dir: Path, arm_keys: list[str]) -> tuple[np.ndarray, np.ndarray] | None:
    init_path = data_dir / "init_pose.json"
    if not init_path.exists():
        return None
    with init_path.open("r", encoding="utf-8") as f:
        init_obj = json.load(f)

    pose = None
    if isinstance(init_obj, dict):
        for key in arm_keys:
            raw = init_obj.get(key)
            if isinstance(raw, list) and len(raw) >= 7:
                pose = np.asarray(raw[:7], dtype=np.float64)
                break
            if isinstance(raw, dict):
                raw_pose = raw.get("pose")
                if isinstance(raw_pose, list) and len(raw_pose) >= 7:
                    pose = np.asarray(raw_pose[:7], dtype=np.float64)
                    break

    if pose is None:
        return None

    return pose[:3], pose[3:7]


def _resolve_arm_stride(action_dim: int) -> int:
    if action_dim % 10 == 0:
        return 10
    if action_dim % 8 == 0:
        return 8
    if action_dim % 7 == 0:
        return 7
    return action_dim


def _select_episodes(data_dir: Path, episode: str) -> list[Path]:
    if episode:
        ep = data_dir / episode
        if not ep.exists():
            raise FileNotFoundError(f"Episode not found: {ep}")
        return [ep]

    episodes = sorted(
        [p for p in data_dir.iterdir() if p.is_dir() and p.name.startswith("episode_")]
    )
    if not episodes:
        raise FileNotFoundError(f"No episode_xxxx directories found under: {data_dir}")
    return episodes


def _extract_color_paths(item: dict, ep_path: Path) -> dict[str, Path]:
    out = {}
    colors = item.get("colors")
    if not isinstance(colors, dict):
        return out
    for cam_key, cam_obj in colors.items():
        if not isinstance(cam_obj, dict):
            continue
        rel_path = cam_obj.get("path")
        if not isinstance(rel_path, str) or not rel_path:
            continue
        full_path = ep_path / rel_path
        if full_path.exists():
            out[cam_key] = full_path
    return out


def _load_image_rgba(img_path: Path, downsample: int, alpha: float) -> np.ndarray | None:
    try:
        img = plt.imread(img_path)
    except Exception:
        return None

    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    if img.ndim != 3:
        return None

    if img.dtype.kind in ("u", "i"):
        img = img.astype(np.float32) / 255.0
    else:
        img = img.astype(np.float32)

    if img.shape[2] >= 4:
        rgba = img[:, :, :4]
    elif img.shape[2] == 3:
        ones = np.ones((*img.shape[:2], 1), dtype=np.float32)
        rgba = np.concatenate([img, ones], axis=2)
    else:
        return None

    step = max(1, downsample)
    rgba = rgba[::step, ::step, :]
    if rgba.shape[0] < 2 or rgba.shape[1] < 2:
        return None

    rgba = np.clip(rgba, 0.0, 1.0)
    rgba[:, :, 3] = np.clip(rgba[:, :, 3] * alpha, 0.0, 1.0)
    return rgba


def _draw_start_images(
    ax,
    chunks: list[dict],
    all_points_arr: np.ndarray,
    image_key: str,
    image_scale_ratio: float,
    image_lift_ratio: float,
    image_downsample: int,
    image_alpha: float,
) -> int:
    mins = all_points_arr.min(axis=0)
    maxs = all_points_arr.max(axis=0)
    span = float(np.max(maxs - mins))
    if span <= 0:
        span = 1e-3

    half_w = max(1e-4, span * image_scale_ratio * 0.5)
    lift = max(1e-4, span * image_lift_ratio)

    img_cache: dict[Path, np.ndarray | None] = {}
    rendered = 0

    for chunk in chunks:
        start_images = chunk.get("start_images", {})
        img_path = start_images.get(image_key)
        if img_path is None:
            continue

        if img_path not in img_cache:
            img_cache[img_path] = _load_image_rgba(
                img_path=img_path,
                downsample=image_downsample,
                alpha=image_alpha,
            )
        rgba = img_cache[img_path]
        if rgba is None:
            continue

        h, w = rgba.shape[:2]
        half_h = half_w * (h / max(1, w))
        x0, y0, z0 = chunk["points"][0]
        z_img = z0 + lift

        xs = np.linspace(x0 - half_w, x0 + half_w, w)
        ys = np.linspace(y0 - half_h, y0 + half_h, h)
        xx, yy = np.meshgrid(xs, ys)
        zz = np.full_like(xx, z_img)

        ax.plot_surface(
            xx,
            yy,
            zz,
            rstride=1,
            cstride=1,
            facecolors=rgba,
            shade=False,
            linewidth=0.0,
            antialiased=False,
        )
        ax.plot([x0, x0], [y0, y0], [z0, z_img], color="gray", lw=0.7, alpha=0.45)
        rendered += 1

    return rendered


def _load_chunks(
    data_dir: Path,
    episode: str,
    arm_index: int,
    arm_stride: int | None,
    point_stride: int,
    compose_init_pose: bool,
) -> tuple[list[dict], list[dict]]:
    episodes = _select_episodes(data_dir, episode)
    if arm_index == 0:
        arm_keys = ["single", "left"]
    elif arm_index == 1:
        arm_keys = ["right"]
    else:
        arm_keys = [f"arm_{arm_index}"]

    init_pose = _load_init_pose(data_dir, arm_keys) if compose_init_pose else None

    chunks = []
    gap_stats = []

    for ep_path in episodes:
        json_path = ep_path / "data.json"
        if not json_path.exists():
            continue

        with json_path.open("r", encoding="utf-8") as f:
            obj = json.load(f)

        data = obj.get("data", [])
        ep_chunks = []

        for item in data:
            actions = item.get("actions")
            if not isinstance(actions, list) or len(actions) == 0:
                continue

            arr = np.asarray(actions, dtype=np.float64)
            if arr.ndim != 2 or arr.shape[1] < 3:
                continue

            stride = arm_stride if arm_stride is not None else _resolve_arm_stride(arr.shape[1])
            base = arm_index * stride
            if base + 3 > arr.shape[1]:
                continue

            pts = arr[:, base : base + 3]
            pts = pts[:: max(1, point_stride)]
            if pts.shape[0] == 0:
                continue

            if init_pose is not None:
                t0, q0 = init_pose
                rot0 = R.from_quat(q0)
                pts = t0[None, :] + rot0.apply(pts)

            ep_chunks.append(
                {
                    "points": pts,
                    "start_images": _extract_color_paths(item=item, ep_path=ep_path),
                }
            )

        for idx, entry in enumerate(ep_chunks):
            chunks.append(
                {
                    "episode": ep_path.name,
                    "local_chunk_idx": idx,
                    "points": entry["points"],
                    "start_images": entry["start_images"],
                }
            )

        for idx in range(len(ep_chunks) - 1):
            p_end = ep_chunks[idx]["points"][-1]
            p_next = ep_chunks[idx + 1]["points"][0]
            gap = float(np.linalg.norm(p_end - p_next))
            gap_stats.append(
                {
                    "episode": ep_path.name,
                    "chunk_idx": idx,
                    "next_chunk_idx": idx + 1,
                    "gap_l2": gap,
                }
            )

    return chunks, gap_stats


def visualize_chunks(
    data_dir: Path,
    episode: str,
    arm_index: int,
    arm_stride: int | None,
    point_stride: int,
    max_chunks: int | None,
    random_consecutive_chunks: int | None,
    random_seed: int | None,
    linewidth: float,
    point_size: float,
    endpoint_size: float,
    endpoint_linewidth: float,
    show_jump_links: bool,
    show_start_images: bool,
    start_image_key: str,
    start_image_scale_ratio: float,
    start_image_lift_ratio: float,
    start_image_downsample: int,
    start_image_alpha: float,
    compose_init_pose: bool,
    output: Path | None,
) -> None:
    chunks, _ = _load_chunks(
        data_dir=data_dir,
        episode=episode,
        arm_index=arm_index,
        arm_stride=arm_stride,
        point_stride=point_stride,
        compose_init_pose=compose_init_pose,
    )

    if not chunks:
        raise RuntimeError("No valid action chunks found.")
    else:
        print(f"Loaded total {len(chunks)} chunks, each chunk shape {chunks[0]['points'].shape}.")

    if max_chunks is not None and max_chunks > 0:
        chunks = chunks[:max_chunks]

    if random_consecutive_chunks is not None and random_consecutive_chunks > 0:
        win = min(random_consecutive_chunks, len(chunks))
        rng = random.Random(random_seed)
        start = rng.randint(0, len(chunks) - win)
        chunks = chunks[start : start + win]
        print(
            f"Random consecutive window: start={start}, size={win}, "
            f"seed={random_seed if random_seed is not None else 'None'}"
        )

    n = len(chunks)
    cmap = plt.get_cmap("turbo")
    norm = Normalize(vmin=0, vmax=max(1, n - 1))

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    all_points = []
    endpoint_labels = []
    for idx, chunk in enumerate(chunks):
        pts = chunk["points"]
        color = cmap(norm(idx))

        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=color, lw=linewidth, alpha=0.95)
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], color=[color], s=point_size, alpha=0.85)

        # Mark per-chunk start/end points with larger symbols for easy inspection.
        ax.scatter(
            pts[0, 0],
            pts[0, 1],
            pts[0, 2],
            color=[color],
            marker="^",
            s=endpoint_size,
            edgecolors="none",
            linewidths=endpoint_linewidth,
            zorder=5,
        )
        ax.scatter(
            pts[-1, 0],
            pts[-1, 1],
            pts[-1, 2],
            color=[color],
            marker="x",
            s=endpoint_size,
            linewidths=endpoint_linewidth,
            zorder=6,
        )
        endpoint_labels.append((pts[0], f"S{idx}", color))
        endpoint_labels.append((pts[-1], f"E{idx}", color))

        all_points.append(pts)

    if show_jump_links:
        by_episode = {}
        for c in chunks:
            by_episode.setdefault(c["episode"], []).append(c)
        for _, ep_chunks in by_episode.items():
            ep_chunks = sorted(ep_chunks, key=lambda x: x["local_chunk_idx"])
            for i in range(len(ep_chunks) - 1):
                p0 = ep_chunks[i]["points"][-1]
                p1 = ep_chunks[i + 1]["points"][0]
                ax.plot(
                    [p0[0], p1[0]],
                    [p0[1], p1[1]],
                    [p0[2], p1[2]],
                    color="black",
                    linestyle="--",
                    lw=0.8,
                    alpha=0.45,
                )

    all_points_arr = np.vstack(all_points)
    _set_equal_axes(ax, all_points_arr)
    mins = all_points_arr.min(axis=0)
    maxs = all_points_arr.max(axis=0)
    span = float(np.max(maxs - mins))

    if show_start_images:
        rendered = _draw_start_images(
            ax=ax,
            chunks=chunks,
            all_points_arr=all_points_arr,
            image_key=start_image_key,
            image_scale_ratio=start_image_scale_ratio,
            image_lift_ratio=start_image_lift_ratio,
            image_downsample=max(1, start_image_downsample),
            image_alpha=np.clip(start_image_alpha, 0.0, 1.0),
        )
        print(f"Rendered start images: {rendered} (image-key={start_image_key})")

    text_offset = max(1e-3, span * 0.01)
    for pos, text, color in endpoint_labels:
        ax.text(
            pos[0] + text_offset,
            pos[1] + text_offset,
            pos[2] + text_offset,
            text,
            fontsize=8,
            color=color,
            alpha=0.95,
        )

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.75, pad=0.08)
    cbar.set_label("global chunk index")
    ax.scatter([], [], [], marker="^", s=endpoint_size, color="gray", label="chunk start (S*)")
    ax.scatter([], [], [], marker="x", s=endpoint_size, color="gray", label="chunk end (E*)")
    if show_start_images:
        ax.scatter([], [], [], marker="s", s=40, color="gray", label=f"{start_image_key} image")
    ax.legend(loc="upper left")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Inference Action Chunks in 3D (same color = same chunk)")

    display_gaps = []
    for i in range(len(chunks) - 1):
        c0 = chunks[i]
        c1 = chunks[i + 1]
        if c0["episode"] == c1["episode"] and c1["local_chunk_idx"] == c0["local_chunk_idx"] + 1:
            display_gaps.append(float(np.linalg.norm(c0["points"][-1] - c1["points"][0])))

    if display_gaps:
        gap_values = np.asarray(display_gaps, dtype=np.float64)
        print(
            f"Loaded {n} chunks. Inter-chunk gap L2: mean={gap_values.mean():.6f}, "
            f"max={gap_values.max():.6f}, min={gap_values.min():.6f}"
        )

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=220, bbox_inches="tight")
        print(f"Saved figure to: {output}")
    else:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize action chunks in 3D and separate chunk membership by color."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("dataset/data/fr3_insert_tube_infer/action_chunks_fr3_absolute"),
        help="Directory containing episode_xxxx/data.json",
    )
    parser.add_argument(
        "--episode",
        type=str,
        default="",
        help="Optional single episode name, e.g. episode_0000. If empty, load all episodes.",
    )
    parser.add_argument("--arm-index", type=int, default=0, help="Arm block index in action vector.")
    parser.add_argument(
        "--arm-stride",
        type=int,
        default=None,
        help="Action dims per arm block. Auto-resolve if omitted.",
    )
    parser.add_argument("--point-stride", type=int, default=1, help="Sample every k points within each chunk.")
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        help="Optional cap on rendered chunk count.",
    )
    parser.add_argument(
        "--random-consecutive-chunks",
        type=int,
        default=None,
        help="Randomly pick N consecutive chunks to visualize. If omitted, use all loaded chunks.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="Optional random seed for reproducible random window selection.",
    )
    parser.add_argument("--linewidth", type=float, default=1.0)
    parser.add_argument("--point-size", type=float, default=8.0)
    parser.add_argument(
        "--endpoint-size",
        type=float,
        default=90.0,
        help="Marker size for per-chunk start(^) and end(x) points.",
    )
    parser.add_argument(
        "--endpoint-linewidth",
        type=float,
        default=1.8,
        help="Line width for start/end markers.",
    )
    parser.add_argument(
        "--show-jump-links",
        action="store_true",
        help="Draw dashed links from chunk end to next chunk start.",
    )
    parser.add_argument(
        "--show-start-images",
        action="store_true",
        help="Render a small image plane above each chunk start point.",
    )
    parser.add_argument(
        "--start-image-key",
        type=str,
        default="right_color",
        help="Camera key in data.json.colors used for start-point image (e.g. right_color).",
    )
    parser.add_argument(
        "--start-image-scale-ratio",
        type=float,
        default=0.2,
        help="Image width ratio to trajectory span.",
    )
    parser.add_argument(
        "--start-image-lift-ratio",
        type=float,
        default=0.06,
        help="Vertical lift ratio from start point to image plane.",
    )
    parser.add_argument(
        "--start-image-downsample",
        type=int,
        default=2,
        help="Downsample factor for image rendering to reduce 3D mesh cost.",
    )
    parser.add_argument(
        "--start-image-hd",
        action="store_true",
        help="Use higher image quality for start images (equivalent to downsample=4 unless explicitly set).",
    )
    parser.add_argument(
        "--start-image-alpha",
        type=float,
        default=0.95,
        help="Alpha for image planes in [0,1].",
    )
    parser.add_argument(
        "--compose-init-pose",
        action="store_true",
        help="If init_pose.json exists, compose relative xyz with init pose.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Output image path. If omitted, show window.")

    args = parser.parse_args()

    if not args.data_dir.exists():
        raise FileNotFoundError(f"data-dir not found: {args.data_dir}")

    image_downsample = max(1, args.start_image_downsample)
    if args.start_image_hd and image_downsample > 4:
        image_downsample = 4

    visualize_chunks(
        data_dir=args.data_dir,
        episode=args.episode,
        arm_index=args.arm_index,
        arm_stride=args.arm_stride,
        point_stride=max(1, args.point_stride),
        max_chunks=args.max_chunks,
        random_consecutive_chunks=args.random_consecutive_chunks,
        random_seed=args.random_seed,
        linewidth=args.linewidth,
        point_size=args.point_size,
        endpoint_size=args.endpoint_size,
        endpoint_linewidth=args.endpoint_linewidth,
        show_jump_links=args.show_jump_links,
        show_start_images=args.show_start_images,
        start_image_key=args.start_image_key,
        start_image_scale_ratio=max(1e-4, args.start_image_scale_ratio),
        start_image_lift_ratio=max(1e-4, args.start_image_lift_ratio),
        start_image_downsample=image_downsample,
        start_image_alpha=args.start_image_alpha,
        compose_init_pose=args.compose_init_pose,
        output=args.output,
    )


if __name__ == "__main__":
    main()
