import os, glob
import cv2
import numpy as np

# ===== 경로 =====
FRAMES_DIR = "data/frames"
OUT_PATH = "data/state.npy"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PLAYER_TMPL = os.path.join(BASE_DIR, "..", "image", "player.png")
MONSTER_TMPL = os.path.join(BASE_DIR, "..", "image", "monster.png")

METHOD = cv2.TM_CCOEFF_NORMED
SCORE_MIN = 0.20  # 너무 낮으면 “못 찾음”으로 간주(필요하면 조절)

def match_one(frame_bgr, tmpl_bgr):
    res = cv2.matchTemplate(frame_bgr, tmpl_bgr, METHOD)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    x, y = max_loc
    h, w = tmpl_bgr.shape[:2]
    cx, cy = x + w / 2.0, y + h / 2.0
    return float(max_val), float(cx), float(cy)

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

    # 첫 프레임으로 ROI 크기(정규화 기준) 확인
    first = cv2.imread(frame_paths[0])
    H, W = first.shape[:2]

    states = []
    bad = 0

    for i, p in enumerate(frame_paths):
        frame = cv2.imread(p)

        ps, px, py = match_one(frame, player_t)
        ms, mx, my = match_one(frame, monster_t)

        # 점수 너무 낮으면 bad 카운트(그래도 값은 넣음)
        if ps < SCORE_MIN or ms < SCORE_MIN:
            bad += 1

        # (x,y) 정규화: 0~1
        pxn, pyn = px / W, py / H
        mxn, myn = mx / W, my / H

        # 상대 위치(도움 많이 됨)
        dx, dy = (mx - px) / W, (my - py) / H

        # state = [px, py, mx, my, dx, dy] (모두 정규화)
        states.append([pxn, pyn, mxn, myn, dx, dy])

        if (i + 1) % 500 == 0:
            print(f"processed {i+1}/{len(frame_paths)}")

    states = np.asarray(states, dtype=np.float32)
    np.save(OUT_PATH, states)
    print("saved:", OUT_PATH, states.shape)
    print(f"low-score frames: {bad}/{len(frame_paths)} (SCORE_MIN={SCORE_MIN})")

if __name__ == "__main__":
    main()
