import time
import cv2
import numpy as np
import mss
from ultralytics import YOLO

# ========= 설정 =========
MODEL_PATH = "yolov8n.pt"   # 속도 우선이면 n, 정확도 조금 더면 s
CONF = 0.5

# ========= 전역(ROI 선택용) =========
roi = None                 # (x1, y1, x2, y2) in preview coords
dragging = False
start_pt = (0, 0)
current_pt = (0, 0)

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

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
        # 정렬
        x1, x2 = sorted([x1, x2])
        y1, y2 = sorted([y1, y2])
        # 너무 작은 선택 방지
        if (x2 - x1) >= 10 and (y2 - y1) >= 10:
            roi = (x1, y1, x2, y2)

def pick_roi_from_screen(monitor_index=1, preview_max_width=1200):
    """
    monitor_index:
      1 = 첫 번째 모니터 (mss 기준)
    반환: (region, scale)
      region: {"left","top","width","height"} in REAL screen coords
      scale: preview->real scaling factor
    """
    global roi, dragging, start_pt, current_pt

    roi = None
    dragging = False

    with mss.mss() as sct:
        mon = sct.monitors[monitor_index]  # {"left","top","width","height"}
        grab = sct.grab(mon)
        img = np.array(grab)
        frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        h, w = frame.shape[:2]
        # 미리보기 크기 조정(너무 큰 화면이면 줄여서 드래그하기 편하게)
        scale = 1.0
        if w > preview_max_width:
            scale = preview_max_width / w
            frame_preview = cv2.resize(frame, (int(w*scale), int(h*scale)))
        else:
            frame_preview = frame

        win = "Select ROI (drag) | ENTER: confirm | ESC: cancel"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(win, on_mouse)

        while True:
            canvas = frame_preview.copy()

            # 드래그 중 박스 표시
            if dragging:
                x1, y1 = start_pt
                x2, y2 = current_pt
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 확정된 ROI 표시
            if roi is not None:
                x1, y1, x2, y2 = roi
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(canvas, "ROI selected. Press ENTER to confirm.",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
            else:
                cv2.putText(canvas, "Drag to select ROI",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

            cv2.imshow(win, canvas)
            key = cv2.waitKey(16) & 0xFF

            # ENTER 확정
            if key == 13 and roi is not None:
                cv2.destroyWindow(win)
                # preview 좌표 -> real screen 좌표
                x1, y1, x2, y2 = roi
                inv = 1.0 / scale
                rx1 = int(x1 * inv)
                ry1 = int(y1 * inv)
                rx2 = int(x2 * inv)
                ry2 = int(y2 * inv)

                # 모니터 기준 오프셋 반영
                region = {
                    "left": mon["left"] + rx1,
                    "top":  mon["top"]  + ry1,
                    "width":  max(1, rx2 - rx1),
                    "height": max(1, ry2 - ry1),
                }
                return region, scale

            # ESC 취소
            if key == 27:
                cv2.destroyWindow(win)
                return None, None

def run_screen_yolo(region):
    model = YOLO(MODEL_PATH)

    with mss.mss() as sct:
        while True:
            t0 = time.time()

            img = np.array(sct.grab(region))
            frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

            results = model(frame, conf=CONF, verbose=False)
            annotated = results[0].plot()

            fps = 1.0 / max(1e-6, (time.time() - t0))
            cv2.putText(annotated, f"FPS: {fps:.1f}  (r: reselect, q: quit)",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

            cv2.imshow("YOLO Screen ROI", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                # 재선택
                cv2.destroyWindow("YOLO Screen ROI")
                return "reselect"

    cv2.destroyAllWindows()
    return "quit"

def main():
    monitor_index = 1  # 첫 번째 모니터. 멀티 모니터면 2,3... 바꿔도 됨.
    while True:
        region, _ = pick_roi_from_screen(monitor_index=monitor_index)
        if region is None:
            print("ROI selection canceled.")
            break

        status = run_screen_yolo(region)
        if status == "quit":
            break

if __name__ == "__main__":
    main()
