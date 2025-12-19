# record.py
import os, time
import cv2
import numpy as np
import mss
from pynput import keyboard

# ====== 설정 ======
SAVE_DIR = "data"
FRAMES_DIR = os.path.join(SAVE_DIR, "frames")
FPS = 20                 # 기록 주기
MAX_SAMPLES = 5000       # 최대 저장 샘플 수
PREVIEW_MAX_WIDTH = 1200

# ====== 키 상태 추적 ======
pressed = set()

def on_press(key):
    pressed.add(key)

def on_release(key):
    pressed.discard(key)

def get_action_vec():
    # [UP, DOWN, LEFT, RIGHT, ALT, SHIFT]
    up = 1 if keyboard.Key.up in pressed else 0
    down = 1 if keyboard.Key.down in pressed else 0
    left = 1 if keyboard.Key.left in pressed else 0
    right = 1 if keyboard.Key.right in pressed else 0

    alt = 1 if (keyboard.Key.alt in pressed or keyboard.Key.alt_l in pressed or keyboard.Key.alt_r in pressed) else 0
    shift = 1 if (keyboard.Key.shift in pressed or keyboard.Key.shift_r in pressed) else 0

    return np.array([up, down, left, right, alt, shift], dtype=np.uint8)

# ====== ROI 선택 UI ======
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
    global roi, dragging, start_pt, current_pt
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

            # ENTER 확정
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

            # ESC 취소
            if key == 27:
                cv2.destroyWindow(win)
                return None

# ====== 기록 루프 ======
def main():
    os.makedirs(FRAMES_DIR, exist_ok=True)

    listener = None
    labels = []
    dt = 1.0 / FPS

    try:
        # 1) 키 리스너 시작
        listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        listener.start()

        # 2) ROI 선택
        monitor_index = 1
        region = pick_roi_from_screen(monitor_index=monitor_index, preview_max_width=PREVIEW_MAX_WIDTH)
        if region is None:
            print("ROI selection canceled.")
            return

        # 3) 기록
        print("Recording...  [q] stop (window X / exception also saves)")
        with mss.mss() as sct:
            for i in range(MAX_SAMPLES):
                t0 = time.time()

                img = np.array(sct.grab(region))
                frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

                action = get_action_vec()
                labels.append(action)

                out_path = os.path.join(FRAMES_DIR, f"{i:06d}.jpg")
                cv2.imwrite(out_path, frame)

                # 프리뷰 표시
                cv2.putText(frame, f"{i} action={action.tolist()} (q: stop)",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("Recording ROI", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

                elapsed = time.time() - t0
                if elapsed < dt:
                    time.sleep(dt - elapsed)

    finally:
        # ✅ 어떤 종료든 저장 시도
        try:
            if len(labels) > 0:
                labels_arr = np.stack(labels, axis=0)  # (N, 5)
                os.makedirs(SAVE_DIR, exist_ok=True)
                np.save(os.path.join(SAVE_DIR, "labels.npy"), labels_arr)
                print("labels.npy saved:", labels_arr.shape)
            else:
                print("No labels collected (0 samples). labels.npy not saved.")
        except Exception as e:
            print("Failed to save labels.npy:", repr(e))

        if listener is not None:
            try:
                listener.stop()
            except Exception:
                pass

        cv2.destroyAllWindows()
        print(f"Done. Frames dir: {FRAMES_DIR}")

if __name__ == "__main__":
    main()
