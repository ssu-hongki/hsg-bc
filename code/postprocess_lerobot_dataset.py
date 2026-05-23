import os
import glob
import json
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


SUBGOAL_NAMES = [
    "move_to_source_object",
    "pick_source_object",
    "move_source_to_target",
    "pour_source_into_target",
    "move_source_back",
    "release_source_object",
    "move_to_tool_object",
    "pick_tool_object",
    "move_tool_to_target",
    "stir_target",
    "move_tool_back",
    "release_tool_object",
]

NUM_SUBGOALS = 12

ARUCO_ID_TO_OBJECT = {
    1: "beaker_A",
    2: "beaker_B",
    3: "stick",
}

OBJECT_ORDER = ["beaker_A", "beaker_B", "stick"]


def find_column(columns, candidates):
    for c in candidates:
        if c in columns:
            return c
    return None


def to_array(x):
    if isinstance(x, np.ndarray):
        return x.astype(np.float32)

    if isinstance(x, list):
        return np.asarray(x, dtype=np.float32)

    if isinstance(x, tuple):
        return np.asarray(x, dtype=np.float32)

    return np.asarray(x, dtype=np.float32)


def stack_vector_column(series):
    values = []
    for x in series:
        values.append(to_array(x))
    return np.stack(values, axis=0).astype(np.float32)


def get_episode_id(df, path, file_idx):
    if "episode_index" in df.columns:
        return str(int(df["episode_index"].iloc[0]))

    name = Path(path).stem
    nums = "".join([ch if ch.isdigit() else " " for ch in name]).split()
    if len(nums) > 0:
        return str(int(nums[-1]))

    return str(file_idx)


def get_frame_indices(df):
    if "frame_index" in df.columns:
        return np.asarray(df["frame_index"], dtype=np.int64)

    return np.arange(len(df), dtype=np.int64)


def build_subgoals_for_episode(episode_id, frame_indices, segments):
    subgoals = np.zeros((len(frame_indices), NUM_SUBGOALS), dtype=np.float32)
    subgoal_ids = np.full((len(frame_indices),), -1, dtype=np.int64)

    if episode_id not in segments:
        return subgoals, subgoal_ids

    for seg in segments[episode_id]:
        start = int(seg["start"])
        end = int(seg["end"])
        label = int(seg["label"])

        if label < 0 or label >= NUM_SUBGOALS:
            continue

        mask = (frame_indices >= start) & (frame_indices < end)
        subgoals[mask, label] = 1.0
        subgoal_ids[mask] = label

    return subgoals, subgoal_ids


def init_aruco_detector():
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, params)
    return detector


def image_from_cell(cell, data_root):
    if isinstance(cell, np.ndarray):
        img = cell
        if img.ndim == 3:
            return img
        return None

    if isinstance(cell, str):
        path = Path(cell)
        if not path.is_absolute():
            path = Path(data_root) / cell
        if path.exists():
            img = cv2.imread(str(path))
            return img
        return None

    if isinstance(cell, dict):
        for key in ["path", "image_path", "file", "filename"]:
            if key in cell:
                path = Path(cell[key])
                if not path.is_absolute():
                    path = Path(data_root) / cell[key]
                if path.exists():
                    img = cv2.imread(str(path))
                    return img

    return None


def extract_aruco_visual_state(frame, detector):
    visual_state = {
        name: [-1.0, -1.0, 0.0]
        for name in OBJECT_ORDER
    }

    if frame is None:
        out = []
        for name in OBJECT_ORDER:
            out.extend(visual_state[name])
        return np.asarray(out, dtype=np.float32)

    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    corners, ids, _ = detector.detectMarkers(gray)

    if ids is not None:
        ids = ids.flatten()

        for corner, marker_id in zip(corners, ids):
            marker_id = int(marker_id)

            if marker_id not in ARUCO_ID_TO_OBJECT:
                continue

            name = ARUCO_ID_TO_OBJECT[marker_id]

            pts = corner.reshape(4, 2)
            cx, cy = pts.mean(axis=0)

            x_norm = float(cx / w)
            y_norm = float(cy / h)

            visual_state[name] = [x_norm, y_norm, 1.0]

    out = []
    for name in OBJECT_ORDER:
        out.extend(visual_state[name])

    return np.asarray(out, dtype=np.float32)


def extract_vision_features(df, image_col, data_root, use_aruco):
    if not use_aruco or image_col is None:
        return np.zeros((len(df), len(OBJECT_ORDER) * 3), dtype=np.float32)

    detector = init_aruco_detector()

    features = []
    for i in range(len(df)):
        frame = image_from_cell(df[image_col].iloc[i], data_root)
        visual = extract_aruco_visual_state(frame, detector)
        features.append(visual)

        if (i + 1) % 500 == 0:
            print(f"  vision processed: {i+1}/{len(df)}")

    return np.stack(features, axis=0).astype(np.float32)


def inspect_dataset(data_root):
    parquet_files = sorted(glob.glob(os.path.join(data_root, "**", "*.parquet"), recursive=True))
    print("num parquet files:", len(parquet_files))

    if len(parquet_files) == 0:
        return

    p = parquet_files[0]
    df = pd.read_parquet(p)

    print("sample file:", p)
    print("shape:", df.shape)
    print("columns:")
    for c in df.columns:
        print(" -", c)

    for key in ["observation.state", "action", "episode_index", "frame_index", "timestamp"]:
        if key in df.columns:
            print(f"\n[{key}] example:")
            print(df[key].iloc[0])


def process_dataset(data_root, output_dir, segment_path, use_aruco=True):
    data_root = str(data_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(segment_path, "r", encoding="utf-8") as f:
        segments = json.load(f)

    parquet_files = sorted(glob.glob(os.path.join(data_root, "**", "*.parquet"), recursive=True))

    if len(parquet_files) == 0:
        raise FileNotFoundError(f"No parquet files found in {data_root}")

    print("num parquet files:", len(parquet_files))

    for file_idx, path in enumerate(parquet_files):
        print("\nprocessing:", path)

        df = pd.read_parquet(path)

        state_col = find_column(
            df.columns,
            [
                "observation.state",
                "observation_state",
                "state",
                "observation.joint_positions",
                "joint_positions",
            ],
        )

        action_col = find_column(
            df.columns,
            [
                "action",
                "actions",
            ],
        )

        image_col = find_column(
            df.columns,
            [
                "observation.images.cam_top",
                "observation.images.top",
                "observation.images.camera_top",
                "observation.images.cam_follower",
                "observation.image",
                "image",
            ],
        )

        if state_col is None:
            print("skip: state column not found")
            print(df.columns)
            continue

        if action_col is None:
            print("skip: action column not found")
            print(df.columns)
            continue

        episode_id = get_episode_id(df, path, file_idx)
        frame_indices = get_frame_indices(df)

        joint_states = stack_vector_column(df[state_col])
        actions = stack_vector_column(df[action_col])

        vision_features = extract_vision_features(
            df=df,
            image_col=image_col,
            data_root=data_root,
            use_aruco=use_aruco,
        )

        subgoals, subgoal_ids = build_subgoals_for_episode(
            episode_id=episode_id,
            frame_indices=frame_indices,
            segments=segments,
        )

        n = min(len(joint_states), len(actions), len(vision_features), len(subgoals))

        joint_states = joint_states[:n]
        actions = actions[:n]
        vision_features = vision_features[:n]
        subgoals = subgoals[:n]
        subgoal_ids = subgoal_ids[:n]
        frame_indices = frame_indices[:n]

        states = np.concatenate([joint_states, vision_features], axis=1).astype(np.float32)

        valid_mask = subgoal_ids >= 0

        states = states[valid_mask]
        joint_states = joint_states[valid_mask]
        vision_features = vision_features[valid_mask]
        subgoals = subgoals[valid_mask]
        subgoal_ids = subgoal_ids[valid_mask]
        actions = actions[valid_mask]
        frame_indices = frame_indices[valid_mask]

        out_path = output_dir / f"episode_{int(episode_id):06d}_processed.npz"

        np.savez_compressed(
            out_path,
            states=states.astype(np.float32),
            joint_states=joint_states.astype(np.float32),
            vision_features=vision_features.astype(np.float32),
            subgoals=subgoals.astype(np.float32),
            subgoal_ids=subgoal_ids.astype(np.int64),
            actions=actions.astype(np.float32),
            frame_indices=frame_indices.astype(np.int64),
            subgoal_names=np.asarray(SUBGOAL_NAMES),
            object_order=np.asarray(OBJECT_ORDER),
            source_parquet=str(path),
        )

        print("saved:", out_path)
        print("  states:", states.shape)
        print("  joint_states:", joint_states.shape)
        print("  vision_features:", vision_features.shape)
        print("  subgoals:", subgoals.shape)
        print("  actions:", actions.shape)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="data_unified")
    parser.add_argument("--output-dir", type=str, default="data_processed")
    parser.add_argument("--segments", type=str, default="data_unified/subgoal_segments.json")
    parser.add_argument("--inspect", action="store_true")
    parser.add_argument("--no-aruco", action="store_true")

    args = parser.parse_args()

    if args.inspect:
        inspect_dataset(args.data_root)
        return

    process_dataset(
        data_root=args.data_root,
        output_dir=args.output_dir,
        segment_path=args.segments,
        use_aruco=not args.no_aruco,
    )


if __name__ == "__main__":
    main()