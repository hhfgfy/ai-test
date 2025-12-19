import os
import time
import cv2
import numpy as np
import mss

# 템플릿 경로 (find_positions.py에서 잘 되게 만든 방식 그대로 쓰면 됨)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PLAYER_TMPL = os.path.join(BASE_DIR, "..", "image", "player.png")
MONSTER_TMPL = os.path.join(BASE_DIR, "..", "image", "monster.png")

PREVIEW_MAX_WIDTH = 1200
METHOD = cv2.TM_CCOEFF_NORMED

# ====== ROI 선택 UI (record.py와 동일) ======
roi = None
dragging = False
start_pt = (0, 0)
current_pt = (0, 0)

def on_mouse(event, x, y, flags, param):
    global roi, dragging, start_pt, current_pt
    if event == cv2.EVENT_LBUTTONDOWN:
        dragging = True
        start_pt = (x, y)
        current_pt = (x, y)
        roi = None
    elif event == cv2.EVENT_MOUSEMOVE and dragging:
        current_pt = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        dragging = False
        x1, y1 = start_pt
        x2, y2 = current_pt
        x1, x2 = sorted([x1, x2])
        y1, y2 = sorted([y1, y2])
        if (x2 - x1) >= 10 and (y2 - y1) >= 10:
            roi = (x1, y1, x2, y2)

def pick_roi_from_screen(monitor_index=1, preview_max_width=1200):
    global roi, dragging
    roi = None
    dragging = False

    with mss.mss() as sct:
        mon = sct.monitors[monitor_index]
        grab = sct.grab(mon)
        img = np.array(grab)
        frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        h, w = frame.shape[:2]
        scale = 1.0
        if w > preview_max_width:
            scale = preview_max_width / w
            frame_preview = cv2.resize(frame, (int(w * scale), int(h * scale)))
        else:
            frame_preview = frame

        win = "Select ROI (drag) | ENTER: confirm | ESC: cancel"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(win, on_mouse)

        while True:
            canvas = frame_preview.copy()

            if dragging:
                x1, y1 = start_pt
                x2, y2 = current_pt
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)

            if roi is not None:
                x1, y1, x2, y2 = roi
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(canvas, "ROI selected. Press ENTER to confirm.",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            else:
                cv2.putText(canvas, "Drag to select ROI",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            cv2.imshow(win, canvas)
            key = cv2.waitKey(16) & 0xFF

            if key == 13 and roi is not None:
                cv2.destroyWindow(win)
                x1, y1, x2, y2 = roi
                inv = 1.0 / scale
                rx1 = int(x1 * inv); ry1 = int(y1 * inv)
                rx2 = int(x2 * inv); ry2 = int(y2 * inv)

                region = {
                    "left": mon["left"] + rx1,
                    "top":  mon["top"]  + ry1,
                    "width":  max(1, rx2 - rx1),
                    "height": max(1, ry2 - ry1),
                }
                return region

            if key == 27:
                cv2.destroyWindow(win)
                return None

def match_one(frame_bgr, tmpl_bgr):
    res = cv2.matchTemplate(frame_bgr, tmpl_bgr, METHOD)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    x, y = max_loc
    h, w = tmpl_bgr.shape[:2]
    return max_val, (x, y, w, h)

def main():
    player_t = cv2.imread(PLAYER_TMPL)
    monster_t = cv2.imread(MONSTER_TMPL)
    if player_t is None or monster_t is None:
        print("Template load failed.")
        print("PLAYER_TMPL:", PLAYER_TMPL)
        print("MONSTER_TMPL:", MONSTER_TMPL)
        return

    region = pick_roi_from_screen(monitor_index=1, preview_max_width=PREVIEW_MAX_WIDTH)
    if region is None:
        print("ROI selection canceled.")
        return

    with mss.mss() as sct:
        while True:
            img = np.array(sct.grab(region))
            frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

            p_score, (px, py, pw, ph) = match_one(frame, player_t)
            m_score, (mx, my, mw, mh) = match_one(frame, monster_t)

            cv2.rectangle(frame, (px, py), (px+pw, py+ph), (0,255,0), 2)
            cv2.putText(frame, f"PLAYER {p_score:.2f}", (px, max(0, py-5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

            cv2.rectangle(frame, (mx, my), (mx+mw, my+mh), (0,0,255), 2)
            cv2.putText(frame, f"MONSTER {m_score:.2f}", (mx, max(0, my-5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

            cv2.putText(frame, "q: quit", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1)

            cv2.imshow("Live Match (template)", frame)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
