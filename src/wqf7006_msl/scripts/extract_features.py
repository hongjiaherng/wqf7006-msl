import argparse
import os
import warnings
from multiprocessing import Pool

import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm

# -------------------------
# MediaPipe utilities
# -------------------------

mp_holistic = mp.solutions.holistic


def extract_keypoints(results):
    pose = (
        np.array(
            [
                [res.x, res.y, res.z, res.visibility]
                for res in results.pose_landmarks.landmark
            ]
        ).flatten()
        if results.pose_landmarks
        else np.zeros(33 * 4)
    )

    lh = (
        np.array(
            [[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]
        ).flatten()
        if results.left_hand_landmarks
        else np.zeros(21 * 3)
    )

    rh = (
        np.array(
            [[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]
        ).flatten()
        if results.right_hand_landmarks
        else np.zeros(21 * 3)
    )

    return np.concatenate([pose, lh, rh])


# -------------------------
# Video processing
# -------------------------


def _already_extracted(output_path):
    """Check if the output file already exists."""
    return os.path.isfile(output_path)


def _process_video_first(video_path, output_path, num_frames):
    if _already_extracted(output_path):
        # Load existing file to get the number of frames
        existing = np.load(output_path)
        return existing.shape[0]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    warning_path = os.path.join(
        os.path.dirname(os.path.dirname(output_path)), "warning.txt"
    )

    cap = cv2.VideoCapture(video_path)
    keypoints_list = []
    frame_idx = 0

    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as holistic:
        while cap.isOpened() and len(keypoints_list) < num_frames:
            ret, frame = cap.read()
            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = holistic.process(image)

            if results.left_hand_landmarks or results.right_hand_landmarks:
                keypoints = extract_keypoints(results)
                keypoints_list.append(keypoints)

            frame_idx += 1

    cap.release()

    if len(keypoints_list) == 0 and frame_idx > 0:
        warnings.warn(f"No keypoints found in {video_path}, no npy file will be saved.")
        with open(warning_path, "a") as f:
            f.write(f"No keypoints found in {video_path}, no npy file will be saved.\n")
            f.flush()
        return 0

    elif len(keypoints_list) == 0 and frame_idx == 0:
        warnings.warn(f"Invalid video: {video_path}, no npy file will be saved.")
        with open(warning_path, "a") as f:
            f.write(f"Invalid video: {video_path}, no npy file will be saved.\n")
            f.flush()
        return 0

    # Stack all frames into a single array: (num_frames, feature_dim)
    keypoints_array = np.stack(keypoints_list)

    # Save as single file
    np.save(output_path, keypoints_array)
    return keypoints_array.shape[0]


def _process_video_uniform(video_path, output_path, num_frames):
    if _already_extracted(output_path):
        # Load existing file to get the number of frames
        existing = np.load(output_path)
        return existing.shape[0]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    warning_path = os.path.join(
        os.path.dirname(os.path.dirname(output_path)), "warning.txt"
    )

    cap = cv2.VideoCapture(video_path)
    keypoints_list = []
    frame_idx = 0

    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = holistic.process(image)

            if results.left_hand_landmarks or results.right_hand_landmarks:
                keypoints = extract_keypoints(results)
                keypoints_list.append(keypoints)

            frame_idx += 1

    cap.release()

    if len(keypoints_list) == 0 and frame_idx > 0:
        warnings.warn(f"No keypoints found in {video_path}, no npy file will be saved.")
        with open(warning_path, "a") as f:
            f.write(f"No keypoints found in {video_path}, no npy file will be saved.\n")
            f.flush()
        return 0

    elif len(keypoints_list) == 0 and frame_idx == 0:
        warnings.warn(f"Invalid video: {video_path}, no npy file will be saved.")
        with open(warning_path, "a") as f:
            f.write(f"Invalid video: {video_path}, no npy file will be saved.\n")
            f.flush()
        return 0

    # Uniform sampling AFTER extraction
    if len(keypoints_list) >= num_frames:
        idx = np.linspace(0, len(keypoints_list) - 1, num_frames).astype(int)
        keypoints_list = [keypoints_list[i] for i in idx]

    # Stack all frames into a single array: (num_frames, feature_dim)
    keypoints_array = np.stack(keypoints_list)

    # Save as single file
    np.save(output_path, keypoints_array)
    return keypoints_array.shape[0]


def process_video(video_path, output_path, sampling, num_frames):
    if sampling == "first":
        return _process_video_first(video_path, output_path, num_frames)
    elif sampling == "uniform":
        return _process_video_uniform(video_path, output_path, num_frames)
    else:
        raise ValueError(f"Unknown sampling: {sampling}")


# -------------------------
# Multiprocessing helper
# -------------------------


def _worker_star(task):
    """
    Top-level helper for Windows multiprocessing.
    """
    video_path, output_path, sampling, num_frames = task
    return process_video(video_path, output_path, sampling, num_frames)


# -------------------------
# Main
# -------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Extract MediaPipe features from BIM videos"
    )

    parser.add_argument("--video-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--sampling",
        choices=["first", "uniform"],
        default="first",
    )
    parser.add_argument("--num-frames", type=int, default=30)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--gloss",
        nargs="+",
        help="List of glosses to process (must all exist)",
    )

    args = parser.parse_args()

    output_root = os.path.join(
        args.output_root,
        f"{args.sampling}_{args.num_frames}",
    )

    # ---------- validate glosses ----------
    available_glosses = sorted(
        d
        for d in os.listdir(args.video_root)
        if os.path.isdir(os.path.join(args.video_root, d))
    )

    if args.gloss:
        missing = sorted(set(args.gloss) - set(available_glosses))
        if missing:
            raise ValueError(
                f"Invalid gloss(es): {missing}\nAvailable glosses: {available_glosses}"
            )
        glosses_to_process = args.gloss
    else:
        glosses_to_process = available_glosses

    # ---------- build tasks ----------
    tasks = []

    for gloss in glosses_to_process:
        gloss_dir = os.path.join(args.video_root, gloss)
        output_gloss_dir = os.path.join(output_root, gloss)
        os.makedirs(output_gloss_dir, exist_ok=True)

        for video in os.listdir(gloss_dir):
            if not video.endswith(".mp4"):
                continue

            # Change output from directory to single .npy file
            video_name = os.path.splitext(video)[0]  # Remove .mp4 extension
            output_path = os.path.join(output_gloss_dir, f"{video_name}.npy")

            tasks.append(
                (
                    os.path.join(gloss_dir, video),
                    output_path,
                    args.sampling,
                    args.num_frames,
                )
            )

    # ---------- multiprocessing ----------
    with Pool(args.num_workers) as pool:
        for _ in tqdm(
            pool.imap_unordered(
                _worker_star,
                tasks,
                chunksize=4,
            ),
            total=len(tasks),
        ):
            pass


if __name__ == "__main__":
    main()
