#!/usr/bin/env python3
import argparse
import json
import os
import re
import sys
from typing import Dict, List, Tuple

import cv2

IMAGE_EXTS = (".jpg", ".jpeg", ".png")
FILE_RE = re.compile(r"^(?P<idx>\d+)_(?P<key>.+)\.(?P<ext>jpg|jpeg|png)$", re.IGNORECASE)


def discover_episode_dirs(task_dir: str) -> list[str]:
    task_dir = os.path.abspath(task_dir)
    episodes = sorted(
        d for d in os.listdir(task_dir)
        if d.startswith("episode_") and os.path.isdir(os.path.join(task_dir, d))
    )
    return [os.path.join(task_dir, ep) for ep in episodes]


def load_fps(episode_dir: str) -> float | None:
    json_path = os.path.join(episode_dir, "data.json")
    if not os.path.isfile(json_path):
        return None
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    info = data.get("info", {})
    image_info = info.get("image", {})
    fps = image_info.get("fps")
    return float(fps) if fps is not None else None


def group_frames(colors_dir: str) -> tuple[Dict[str, List[Tuple[int, str]]], int]:
    groups: Dict[str, List[Tuple[int, str]]] = {}
    skipped = 0
    for name in os.listdir(colors_dir):
        lower = name.lower()
        if not lower.endswith(IMAGE_EXTS):
            continue
        match = FILE_RE.match(name)
        if not match:
            skipped += 1
            continue
        idx = int(match.group("idx"))
        key = match.group("key")
        groups.setdefault(key, []).append((idx, os.path.join(colors_dir, name)))
    for key in groups:
        groups[key].sort(key=lambda item: item[0])
    return groups, skipped


def read_image(path: str):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def write_video(frames: List[Tuple[int, str]], output_path: str, fps: float) -> None:
    if not frames:
        return
    first = read_image(frames[0][1])
    if first is None:
        print(f"Skip video {output_path}: failed to read {frames[0][1]}")
        return
    height, width = first.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    for _, path in frames:
        img = read_image(path)
        if img is None:
            print(f"Warning: failed to read {path}")
            continue
        if img.shape[:2] != (height, width):
            # Keep a consistent frame size for the encoder.
            img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
        writer.write(img)
    writer.release()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_dir", required=True, help="Task directory containing episode_* folders.")
    parser.add_argument("--save_dir", required=True, help="Directory to store episode videos.")
    parser.add_argument("--fps", type=float, default=30.0, help="Override FPS for videos.")
    args = parser.parse_args()

    if not os.path.isdir(args.task_dir):
        print(f"Task dir not found: {args.task_dir}")
        return 1

    episode_dirs = discover_episode_dirs(args.task_dir)
    if not episode_dirs:
        print(f"No episode_* directories found under: {args.task_dir}")
        return 1

    for episode_dir in episode_dirs:
        episode_name = os.path.basename(episode_dir)
        colors_dir = os.path.join(episode_dir, "colors")
        if not os.path.isdir(colors_dir):
            print(f"Skip {episode_name}: colors dir not found.")
            continue
        fps = args.fps if args.fps is not None else load_fps(episode_dir) or 30.0
        output_root = os.path.join(args.save_dir, episode_name)

        groups, skipped = group_frames(colors_dir)
        if skipped:
            print(f"Skipped {skipped} files in {colors_dir} (name not matched).")
        if not groups:
            print(f"No images found in {colors_dir}")
            continue

        for view_key, frames in groups.items():
            output_path = os.path.join(output_root, f"{view_key}.mp4")
            write_video(frames, output_path, fps)
            print(f"Saved {output_path} ({len(frames)} frames, fps={fps})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
