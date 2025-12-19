import time
import cv2
import numpy as np
import mss
import torch
import torch.nn as nn
import torchvision.models as models

CKPT_PATH = "policy.pt"
IMG_SIZE = 96
OUT_DIM = 6
KEYS = ["UP","DOWN","LEFT","RIGHT","ALT","SHIFT"]

THR_DIR = 0.5   # 방향키(홀드) 표시 임계값
THR_TAP = 0.6   # 탭키(ALT/SHIFT) 표시 임계값
PREVIEW_MAX_WIDTH = 1200

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

# ====== 모델 ======
class ResNet18Policy(nn.Module):
    def __init__(self, out_dim=6):
        super().__init__()
        self.backbone = models.resnet18(weights=None)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, out_dim)

    def forward(self, x):
        return self.backbone(x)

def preprocess_frame(frame_bgr):
    img = cv2.resize(frame_bgr, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0  # (3,H,W)
    return x.unsqueeze(0)  # (1,3,H,W)

def main():
    # 1) 모델 로드
    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    model = ResNet18Policy(out_dim=OUT_DIM)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # 2) ROI 선택
    region = pick_roi_from_screen(monitor_index=1, preview_max_width=PREVIEW_MAX_WIDTH)
    if region is None:
        print("ROI selection canceled.")
        return

    # 3) 실시간 추론 + 오버레이
    with mss.mss() as sct:
        last_t = time.time()
        while True:
            t0 = time.time()
            img = np.array(sct.grab(region))
            frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

            x = preprocess_frame(frame)
            with torch.no_grad():
                prob = torch.sigmoid(model(x))[0].numpy()  # (6,)

            pred = np.zeros((OUT_DIM,), dtype=np.uint8)
            pred[:4] = (prob[:4] >= THR_DIR).astype(np.uint8)
            pred[4:] = (prob[4:] >= THR_TAP).astype(np.uint8)

            fps = 1.0 / max(1e-6, (t0 - last_t))
            last_t = t0

            text = "PRED " + " ".join([f"{k}:{int(pred[i])}({prob[i]:.2f})" for i,k in enumerate(KEYS)])
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,0), 2)
            cv2.putText(frame, f"FPS:{fps:.1f}  (q: quit)", (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)

            cv2.imshow("Live Preview (policy -> overlay only)", frame)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
