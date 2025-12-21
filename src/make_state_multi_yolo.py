import os
import glob
import cv2
import numpy as np
from ultralytics import YOLO

# ===== paths =====
FRAMES_DIR = "data/frames"
OUT_PATH = "data/state_multi_k3_yolo.npy"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_PATH = os.path.join(BASE_DIR, "yolov8n.pt")

# detection config
CONF_MIN = 0.25
IOU_NMS = 0.5

# number of monster slots in the state
MAX_MONSTERS = 3

# class mapping (set these to your trained model)
PLAYER_CLASS_IDS = [0]
MONSTER_CLASS_IDS = [1]
PLAYER_CLASS_NAMES = []
MONSTER_CLASS_NAMES = []


def _build_name_to_id(names):
    if isinstance(names, dict):
        return {v: int(k) for k, v in names.items()}
    return {v: i for i, v in enumerate(names)}


def _resolve_class_ids(model, ids, names):
    out = set(int(x) for x in ids)
    if names:
        name_to_id = _build_name_to_id(model.names)
        for n in names:
            if n in name_to_id:
                out.add(int(name_to_id[n]))
    return out


def _collect_by_class(results, class_ids, conf_min):
    items = []
    if results.boxes is None:
        return items
    for b in results.boxes:
        cls_id = int(b.cls[0])
        conf = float(b.conf[0])
        if cls_id not in class_ids or conf < conf_min:
            continue
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        items.append((conf, cx, cy, (x1, y1, x2, y2)))
    return items


def main():
    frame_paths = sorted(glob.glob(os.path.join(FRAMES_DIR, "*.jpg")))
    if not frame_paths:
        print("No frames found:", FRAMES_DIR)
        return

    model = YOLO(WEIGHTS_PATH)
    player_ids = _resolve_class_ids(model, PLAYER_CLASS_IDS, PLAYER_CLASS_NAMES)
    monster_ids = _resolve_class_ids(model, MONSTER_CLASS_IDS, MONSTER_CLASS_NAMES)

    first = cv2.imread(frame_paths[0])
    H, W = first.shape[:2]

    states = []
    bad = 0

    for i, p in enumerate(frame_paths):
        frame = cv2.imread(p)
        res = model.predict(frame, conf=CONF_MIN, iou=IOU_NMS, verbose=False)[0]

        players = _collect_by_class(res, player_ids, CONF_MIN)
        monsters = _collect_by_class(res, monster_ids, CONF_MIN)

        if not players or not monsters:
            bad += 1

        if players:
            players.sort(key=lambda x: -x[0])
            _, px, py, _ = players[0]
        else:
            px, py = 0.0, 0.0

        if players:
            monsters = sorted(monsters, key=lambda m: (m[1] - px) ** 2 + (m[2] - py) ** 2)
        else:
            monsters = sorted(monsters, key=lambda m: -m[0])

        pxn, pyn = px / W, py / H
        state = [pxn, pyn]

        for k in range(MAX_MONSTERS):
            if k < len(monsters):
                _, mx, my, _ = monsters[k]
                mxn, myn = mx / W, my / H
                dx, dy = (mx - px) / W, (my - py) / H
            else:
                mxn = myn = dx = dy = 0.0
            state.extend([mxn, myn, dx, dy])

        states.append(state)

        if (i + 1) % 500 == 0:
            print(f"processed {i+1}/{len(frame_paths)}")

    states = np.asarray(states, dtype=np.float32)
    np.save(OUT_PATH, states)
    print("saved:", OUT_PATH, states.shape)
    print(f"low-detect frames: {bad}/{len(frame_paths)} (CONF_MIN={CONF_MIN})")
    print("state_dim:", states.shape[1], "MAX_MONSTERS:", MAX_MONSTERS)


if __name__ == "__main__":
    main()
