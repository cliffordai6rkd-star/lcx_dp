      
"""
Build DTW-based pair_info.json from HIROL JSON episodes.

This script aligns each target episode to a set of source episodes using Dynamic Time
Warping (DTW) on chosen HIROL features (e.g., joint states + gripper) and saves
pair_info.json for OT-style sampling.

Example
  python dataset/dtw_pairing/build_pair_info.py \
    --src-dir dataset/data/1212_duo_unitree_bread_n_picking——214ep \
    --tgt-dir dataset/data/1221_duo_unitree_bread_picking_human_114ep \
    --window 80 --stride 2 --top-k 5 \
    --output dataset/dtw/human_target_bread_picking.json

You can also pass a directory to --output (e.g., --output dataset/dtw/). In that case
the script will auto-name the file as:
  - <src>_pair_info.json                 if src==tgt
  - <src>_to_<tgt>_pair_info.json otherwise

Output JSON format (indices are 0-based):
{
  "episode_0001": [
    {
      "demo_name": "episode_0003",          # source episode id (dirname)
      "raw_dtw_dist": 0.0012,                # normalized distance = dDTW / max(||xs||, ||xt||)
      "pairing": {                           # mapping: target index -> list of aligned source indices
        "0": [0],
        "1": [1, 2]
      }
    }
  ]
}
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from typing import Dict, List, Tuple
from dataset.lerobot.reader import RerunEpisodeReader, ActionType, ObservationType

import numpy as np

# Optional: alternative DTW per arXiv:1607.05994 baseline DP with polyhedral metrics
try:
    from utils.dtw_gs16 import dtw_distance as gs16_dtw_distance
except Exception:
    gs16_dtw_distance = None

# Optional: pollen-robotics/dtw backend (supports cdist-accelerated distance)
try:
    from dtw.dtw import dtw as pollen_dtw
    from dtw.dtw import accelerated_dtw as pollen_accel_dtw
    _has_pollen_dtw = True
except Exception:
    pollen_dtw = None
    pollen_accel_dtw = None
    _has_pollen_dtw = False


# -------------------------
# IO utilities
# -------------------------

def _natural_key(s: str) -> Tuple:
    # Natural sort key for episode_0001-like names
    m = re.findall(r"\d+|\D+", s)
    return tuple(int(x) if x.isdigit() else x for x in m)


def list_episode_jsons(root: str, pattern: str = "episode_") -> List[Tuple[str, str]]:
    """List (episode_id, json_path) pairs under HIROL-style directory tree.

    Expected structure: <root>/<episode_xxxx>/data.json
    """
    out = []
    if not os.path.isdir(root):
        raise FileNotFoundError(f"dir not found: {root}")
    for name in os.listdir(root):
        if not name.startswith(pattern):
            continue
        p = os.path.join(root, name, "data.json")
        if os.path.isfile(p):
            out.append((name, p))
    out.sort(key=lambda x: _natural_key(x[0]))
    return out


def load_hirol_episode_features(task_dir, action_type, skip_nums, low_keys, episode_dir,
                                real_robot_tool_sacle, data_type, rot_transform) -> np.ndarray:
    """Extract a T x D feature sequence from a HIROL episode JSON, following HIROL keys.

    Returns:
      np.ndarray of shape (T, D), float32
    """
    reader = RerunEpisodeReader(task_dir, action_type=action_type, observation_type=ObservationType.MASK,
        data_type=data_type, state_keys=low_keys, rotation_transform=rot_transform,
        real_robot_tool_sacle=real_robot_tool_sacle)
    episode_number = int(episode_dir.lstrip("episode_"))
    episode_id = episode_number
    episode_data = reader.return_episode_data(episode_id, skip_nums)
    if episode_data is None or len(episode_data) == 0:
        return None
    feats: List[np.ndarray] = []
    for step_data in episode_data:
        action_data = step_data.get("actions", None)
        if action_data is None:
            continue
        
        vec = []
        for key in low_keys:
            vec.append(action_data[key])
        vec = np.hstack(vec)
        # print(f'vec shape: {vec.shape}')
        
        feats.append(vec)

    if not feats:
        return np.zeros((0, 0), dtype=np.float32)
    arr = np.stack(feats, axis=0)
    return arr.astype(np.float32)


# -------------------------
# Normalization and DTW
# -------------------------

def zscore(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    std = np.where(std < 1e-6, 1.0, std)
    return (x - mean) / std


def compute_norm_stats(seqs: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    cat = np.concatenate([s for s in seqs if s.size > 0], axis=0)
    mean = cat.mean(axis=0, keepdims=True)
    std = cat.std(axis=0, keepdims=True)
    return mean.astype(np.float32), std.astype(np.float32)


def dtw_path(a: np.ndarray, b: np.ndarray, window: int | None) -> Tuple[List[Tuple[int, int]], float]:
    """DTW with optional Sakoe-Chiba band; returns (path, avg_cost)."""
    Na, Nb = len(a), len(b)
    if Na == 0 or Nb == 0:
        return [], float("inf")
    if window is None:
        window = max(Na, Nb)

    INF = 1e18
    DP = np.full((Na + 1, Nb + 1), INF, dtype=np.float64)
    BP = np.zeros((Na + 1, Nb + 1, 2), dtype=np.int32)
    DP[0, 0] = 0.0

    for i in range(1, Na + 1):
        j_start = max(1, i - window)
        j_end = min(Nb, i + window)
        ai = a[i - 1]
        for j in range(j_start, j_end + 1):
            bj = b[j - 1]
            c = float(np.linalg.norm(ai - bj))
            # (i-1,j), (i,j-1), (i-1,j-1)
            idx = np.argmin((DP[i - 1, j], DP[i, j - 1], DP[i - 1, j - 1]))
            if idx == 0:
                prev = (i - 1, j)
            elif idx == 1:
                prev = (i, j - 1)
            else:
                prev = (i - 1, j - 1)
            DP[i, j] = c + DP[prev[0], prev[1]]
            BP[i, j] = prev

    if not np.isfinite(DP[Na, Nb]):
        return [], float("inf")

    # Backtrack
    i, j = Na, Nb
    path: List[Tuple[int, int]] = []
    while i > 0 or j > 0:
        path.append((i - 1, j - 1))
        pi, pj = BP[i, j]
        if pi == i and pj == j:
            break
        i, j = pi, pj
    path.reverse()
    avg_cost = float(DP[Na, Nb] / max(1, len(path)))
    return path, avg_cost


# -------------------------
# Main logic
# -------------------------

def _resolve_output_path(output: str, src_dir: str, tgt_dir: str) -> str:
    """Allow --output to be either a file path or a directory.

    If it's a directory (existing or endswith path sep), auto-name the file:
      - <src>_pair_info.json if src==tgt
      - <src>_to_<tgt>_<label>_pair_info.json otherwise
    """
    # Treat as directory if it exists or explicitly ends with a separator
    if output.endswith(os.sep) or os.path.isdir(output):
        src_name = os.path.basename(os.path.normpath(src_dir))
        tgt_name = os.path.basename(os.path.normpath(tgt_dir))
        if src_name == tgt_name:
            fname = f"{src_name}_pair_info.json"
        else:
            fname = f"{src_name}_to_{tgt_name}_pair_info.json"
        output = os.path.join(output, fname)
    return output

def build_pair_info_from_hirol(
    src_dir: str,
    tgt_dir: str,
    action_type,
    low_keys,
    rot_transform,
    real_robot_tool_scale,
    src_skip,
    tgt_skip,
    src_data_type,
    tgt_data_type,
    output: str,
    top_k: int | None,
    window: int | None,
    stride: int,
    dtw_impl: str = "pollen",
    metric: str = "l2",
    weight_k: float = 10.0,
    weight_tau: float = 0.01,
):
    scr_episodes = os.listdir(src_dir)
    tgt_episodes = os.listdir(tgt_dir)
    
    # Load sequences and compute normalization
    src_seqs = []; tgt_seqs = []
    for eid in scr_episodes:
        data = load_hirol_episode_features(src_dir, action_type, src_skip, low_keys, 
                eid, real_robot_tool_scale, src_data_type, rot_transform)
        if data is not None:
            src_seqs.append((eid, data))
    print(f'finished loading src with {len(src_seqs)} episodes')
    for eid in tgt_episodes:
        data = load_hirol_episode_features(tgt_dir, action_type, tgt_skip, low_keys, 
                eid, real_robot_tool_scale, tgt_data_type, rot_transform)
        if data is not None:
            tgt_seqs.append((eid, data))
    print(f'finished loading tgt with {len(tgt_seqs)} episodes')

    # Stride downsample if needed
    if stride > 1:
        src_seqs = [(eid, s[::stride]) for eid, s in src_seqs]
        tgt_seqs = [(eid, s[::stride]) for eid, s in tgt_seqs]

    mean, std = compute_norm_stats([s for _, s in src_seqs] + [s for _, s in tgt_seqs])

    src_seqs = [(eid, zscore(s, mean, std)) for eid, s in src_seqs]
    tgt_seqs = [(eid, zscore(s, mean, std)) for eid, s in tgt_seqs]

    result: Dict[str, List[Dict]] = {}
    for t_idx, (tgt_id, tgt_seq) in enumerate(tgt_seqs):
        pairs: List[Tuple[str, float, Dict[str, List[int]]]] = []
        for src_id, src_seq in src_seqs:
            # Choose DTW implementation
            if dtw_impl == "orig":
                path, avg_cost = dtw_path(tgt_seq, src_seq, window=window)
            elif dtw_impl == "gs16":
                if gs16_dtw_distance is None:
                    raise RuntimeError("utils.dtw_gs16 not available; cannot use dtw_impl=gs16")
                path, avg_cost = gs16_dtw_distance(tgt_seq, src_seq, metric=metric, window=window)
            elif dtw_impl == "pollen":
                if not _has_pollen_dtw:
                    print("[warn] pollen-robotics/dtw not available; falling back to 'orig' implementation")
                    path, avg_cost = dtw_path(tgt_seq, src_seq, window=window)
                else:
                    # Map metric to cdist string if using accelerated path.
                    metric_map = {"l2": "euclidean", "l1": "cityblock", "linf": "chebyshev"}
                    if window is None:
                        # Use accelerated_dtw (no window support in upstream accelerated_dtw)
                        dist_name = metric_map.get(metric, "euclidean")
                        cum_cost, C, D1, pq = pollen_accel_dtw(tgt_seq, src_seq, dist=dist_name, warp=1)
                        p, q = map(lambda x: np.asarray(x, dtype=np.int64), pq)
                    else:
                        # Use baseline dtw to honor window; pass a Python metric function
                        def _metric(u, v):
                            if metric == "l1":
                                return float(np.sum(np.abs(u - v)))
                            if metric == "linf":
                                return float(np.max(np.abs(u - v)))
                            return float(np.linalg.norm(u - v))

                        # pollen-robotics/dtw requires w >= |len(x) - len(y)|
                        lt, ls = len(tgt_seq), len(src_seq)
                        w_eff = window if window is not None else None
                        if w_eff is None:
                            w_eff = np.inf
                        else:
                            w_eff = int(max(w_eff, abs(lt - ls)))

                        cum_cost, C, D1, pq = pollen_dtw(tgt_seq, src_seq, dist=_metric, warp=1, w=w_eff, s=1.0)
                        p, q = map(lambda x: np.asarray(x, dtype=np.int64), pq)

                    # Build path and avg_cost compatible with our schema
                    path = list(zip(p.tolist(), q.tolist()))
                    avg_cost = float(cum_cost / max(1, len(path)))
            else:
                raise ValueError(f"Unknown dtw_impl: {dtw_impl}")
            if not path or not np.isfinite(avg_cost):
                continue
            pairing: Dict[int, List[int]] = defaultdict(list)
            for (ti, sj) in path:
                pairing[int(ti * stride)].append(int(sj * stride))
            # Normalization: use trajectory vector norms instead of sequence length.
            # Define ||x|| as sqrt(sum_{t,d} x_{t,d}^2) over the normalized sequences used for DTW.
            ddtw = float(avg_cost * max(1, len(path)))
            xs_norm = float(np.linalg.norm(src_seq))
            xt_norm = float(np.linalg.norm(tgt_seq))
            denom = max(xs_norm, xt_norm)
            norm = float(ddtw / denom) if denom > 1e-12 else float("inf")
            # Logistic weight from normalized distance
            weight = float(1.0 / (1.0 + np.exp(weight_k * (norm - weight_tau)))) if np.isfinite(norm) else 0.0
            pairs.append(
                (
                    src_id,
                    float(avg_cost),
                    {str(k): v for k, v in pairing.items()},
                    ddtw,
                    norm,
                    weight,
                )
            )

        # Keep top-k by distance
        pairs.sort(key=lambda x: x[1])
        if top_k is not None and top_k > 0:
            pairs = pairs[:top_k]

        result[tgt_id] = [
            {
                "demo_name": sid,
                "raw_dtw_dist": norm,  # output normalized distance as raw_dtw_dist
                "pairing": pairing,
            }
            for sid, dist, pairing, ddtw, norm, weight in pairs
        ]
        print(f"Built {len(pairs)} pairs for target {tgt_id}")

    # Save once after all targets are processed
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved pair_info to {output}")


def main():
    parser = argparse.ArgumentParser(description="Build DTW pair_info from HIROL JSON episodes")
    parser.add_argument("--src-dir", required=True, help="source episodes root (contains episode_xxxx/data.json)")
    parser.add_argument("--tgt-dir", required=True, help="target episodes root (contains episode_xxxx/data.json)")
    parser.add_argument("--output", required=True, help="output json path or directory")
    parser.add_argument("--top-k", type=int, default=None, help="keep top-K best source demos per target")
    parser.add_argument("--window", type=int, default=None, help="Sakoe-Chiba band half-width; smaller = faster")
    parser.add_argument("--stride", type=int, default=1, help="temporal stride to downsample before DTW")
    parser.add_argument("--limit-src", type=int, default=None, help="limit number of source episodes (debug)")
    parser.add_argument("--limit-tgt", type=int, default=None, help="limit number of target episodes (debug)")
    parser.add_argument(
        "--dtw-impl",
        choices=["pollen", "orig", "gs16"],
        default="pollen",
        help="choose DTW implementation ('pollen' uses pollen-robotics/dtw and falls back to 'orig' if unavailable)",
    )
    parser.add_argument(
        "--metric",
        choices=["l2", "l1", "linf"],
        default="l2",
        help="distance metric (used by gs16 and pollen backends)",
    )
    parser.add_argument("--weight-k", type=float, default=10.0, help="logistic slope k for weight mapping")
    parser.add_argument("--weight-tau", type=float, default=0.01, help="logistic threshold tau for weight mapping")

    args = parser.parse_args()

    out_path = _resolve_output_path(args.output, args.src_dir, args.tgt_dir)

    action_type = ActionType.END_EFFECTOR_POSE
    rot_transform = {"left": [1, 0, 0, 0], "right": [1, 0, 0, 0],
                     "head": [1, 0, 0, 0]}
    low_keys = ["left", "right", "head"]
    real_robot_tool_scale = 90.0
    src_skip = 2; tgt_skip = 6
    src_data_type = "human"; tgt_data_type = "robot"
    build_pair_info_from_hirol(
        src_dir=args.src_dir,
        tgt_dir=args.tgt_dir,
        action_type=action_type,
        low_keys=low_keys,
        rot_transform=rot_transform,
        real_robot_tool_scale=real_robot_tool_scale,
        src_skip=src_skip,
        tgt_skip=tgt_skip,
        src_data_type=src_data_type,
        tgt_data_type=tgt_data_type,
        output=out_path,
        top_k=args.top_k,
        window=args.window,
        stride=args.stride,
        dtw_impl=args.dtw_impl,
        metric=args.metric,
        weight_k=args.weight_k,
        weight_tau=args.weight_tau,
    )


if __name__ == "__main__":
    main()

    