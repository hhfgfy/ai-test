import os
import glob
import cv2
import numpy as np

# ===== paths =====
FRAMES_DIR = "data/frames"
OUT_PATH = "data/state_multi_k3.npy"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PLAYER_TMPL = os.path.join(BASE_DIR, "..", "image", "player.png")
MONSTER_TMPL = os.path.join(BASE_DIR, "..", "image", "monster.png")

METHOD = cv2.TM_CCOEFF_NORMED
SCORE_MIN_PLAYER = 0.20
SCORE_MIN_MONSTER = 0.20

# number of monster slots in the state
MAX_MONSTERS = 3
NMS_IOU = 0.30


def match_one(frame_bgr, tmpl_bgr):
    res = cv2.matchTemplate(frame_bgr, tmpl_bgr, METHOD)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    x, y = max_loc
    h, w = tmpl_bgr.shape[:2]
    cx, cy = x + w / 2.0, y + h / 2.0
    return float(max_val), float(cx), float(cy)


def _nms(boxes, scores, iou_thresh, max_keep=None):
    if len(boxes) == 0:
        return []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if max_keep is not None and len(keep) >= max_keep:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(iou <= iou_thresh)[0]
        order = order[inds + 1]

    return keep


def match_many(frame_bgr, tmpl_bgr, score_min, max_count=None, nms_iou=0.30):
    res = cv2.matchTemplate(frame_bgr, tmpl_bgr, METHOD)
    h, w = tmpl_bgr.shape[:2]

    ys, xs = np.where(res >= score_min)
    if len(xs) == 0:
        return []

    scores = res[ys, xs].astype(np.float32)
    boxes = np.stack([xs, ys, xs + w, ys + h], axis=1).astype(np.float32)

    keep = _nms(boxes, scores, nms_iou, max_keep=max_count)
    out = []
    for i in keep:
        x, y, x2, y2 = boxes[i]
        cx = (x + x2) / 2.0
        cy = (y + y2) / 2.0
        out.append((float(scores[i]), float(cx), float(cy)))

    return out


def main():
    frame_paths = sorted(glob.glob(os.path.join(FRAMES_DIR, "*.jpg")))
    if not frame_paths:
        print("No frames found:", FRAMES_DIR)
        return

    player_t = cv2.imread(PLAYER_TMPL)
    monster_t = cv2.imread(MONSTER_TMPL)
    if player_t is None or monster_t is None:
        print("Template load failed.")
        print("PLAYER_TMPL:", PLAYER_TMPL)
        print("MONSTER_TMPL:", MONSTER_TMPL)
        return

    first = cv2.imread(frame_paths[0])
    H, W = first.shape[:2]

    states = []
    bad = 0

    for i, p in enumerate(frame_paths):
        frame = cv2.imread(p)

        ps, px, py = match_one(frame, player_t)
        monsters = match_many(
            frame,
            monster_t,
            score_min=SCORE_MIN_MONSTER,
            max_count=None,
            nms_iou=NMS_IOU,
        )

        if ps < SCORE_MIN_PLAYER or len(monsters) == 0:
            bad += 1

        if ps >= SCORE_MIN_PLAYER:
            monsters = sorted(
                monsters,
                key=lambda m: (m[1] - px) ** 2 + (m[2] - py) ** 2,
            )
        else:
            monsters = sorted(monsters, key=lambda m: -m[0])

        # (x,y) normalize to 0~1
        pxn, pyn = px / W, py / H

        state = [pxn, pyn]

        for k in range(MAX_MONSTERS):
            if k < len(monsters):
                _, mx, my = monsters[k]
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
    print(f"low-score frames: {bad}/{len(frame_paths)} "
          f"(PLAYER_MIN={SCORE_MIN_PLAYER}, MONSTER_MIN={SCORE_MIN_MONSTER})")
    print("state_dim:", states.shape[1], "MAX_MONSTERS:", MAX_MONSTERS)


if __name__ == "__main__":
    main()
