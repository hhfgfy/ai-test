import os
import time
import cv2
import numpy as np
import mss
import torch
import torch.nn as nn
import torchvision.models as models

# ====== 모델/데이터 설정 ======
CKPT_PATH = "policy_img_state.pt"
IMG_SIZE = 96
OUT_DIM = 6
STATE_DIM = 6
KEYS = ["UP","DOWN","LEFT","RIGHT","ALT","SHIFT"]

THR_DIR = 0.5   # 방향키(홀드) 표시 임계값
THR_TAP = 0.6   # 탭키(ALT/SHIFT) 표시 임계값
PREVIEW_MAX_WIDTH = 1200

# ====== 템플릿 경로 (네 프로젝트 구조에 맞게) ======
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PLAYER_TMPL = os.path.join(BASE_DIR, "..", "image", "player.png")
MONSTER_TMPL = os.path.join(BASE_DIR, "..", "image", "monster.png")
METHOD = cv2.TM_CCOEFF_NORMED

def match_center(frame_bgr, tmpl_bgr):
    res = cv2.matchTemplate(frame_bgr, tmpl_bgr, METHOD)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    x, y = max_loc
    h, w = tmpl_bgr.shape[:2]
    cx, cy = x + w / 2.0, y + h / 2.0
    return float(max_val), float(cx), float(cy), (x, y, w, h)

def make_state_from_frame(frame_bgr, player_t, monster_t):
    H, W = frame_bgr.shape[:2]

    ps, px, py, pbox = match_center(frame_bgr, player_t)
    ms, mx, my, mbox = match_center(frame_bgr, monster_t)

    # 정규화
    pxn, pyn = px / W, py / H
    mxn, myn = mx / W, my / H
    dx, dy = (mx - px) / W, (my - py) / H

    state = np.array([pxn, pyn, mxn, myn, dx, dy], dtype=np.float32)
    return state, (ps, ms), pbox, mbox

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

# ====== 모델 (train_with_state.py와 동일) ======
class ImgStatePolicy(nn.Module):
    def __init__(self, out_dim=6, state_dim=6):
        super().__init__()
        backbone = models.resnet18(weights=None)
        feat_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone

        self.state_mlp = nn.Sequential(
            nn.Linear(state_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
        )

        self.head = nn.Sequential(
            nn.Linear(feat_dim + 64, 128), nn.ReLU(),
            nn.Linear(128, out_dim)
        )

    def forward(self, x_img, x_state):
        f_img = self.backbone(x_img)
        f_state = self.state_mlp(x_state)
        f = torch.cat([f_img, f_state], dim=1)
        return self.head(f)

def preprocess_frame(frame_bgr):
    img = cv2.resize(frame_bgr, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    return x.unsqueeze(0)  # (1,3,H,W)

def main():
    # 템플릿 로드
    player_t = cv2.imread(PLAYER_TMPL)
    monster_t = cv2.imread(MONSTER_TMPL)
    if player_t is None or monster_t is None:
        print("Template load failed.")
        print("PLAYER_TMPL:", PLAYER_TMPL)
        print("MONSTER_TMPL:", MONSTER_TMPL)
        return

    # 모델 로드
    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    model = ImgStatePolicy(out_dim=OUT_DIM, state_dim=STATE_DIM)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # ROI 선택
    region = pick_roi_from_screen(monitor_index=1, preview_max_width=PREVIEW_MAX_WIDTH)
    if region is None:
        print("ROI selection canceled.")
        return

    # 실시간 추론
    with mss.mss() as sct:
        t_prev = time.time()
        while True:
            img = np.array(sct.grab(region))
            frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

            state_np, (ps, ms), pbox, mbox = make_state_from_frame(frame, player_t, monster_t)

            x_img = preprocess_frame(frame)                         # (1,3,96,96)
            x_state = torch.from_numpy(state_np).unsqueeze(0)       # (1,6)

            with torch.no_grad():
                prob = torch.sigmoid(model(x_img, x_state))[0].numpy()

            best_i = int(prob.argmax())
            best_k = KEYS[best_i]
            cv2.putText(frame, f"BEST: {best_k} ({prob[best_i]:.2f})",
            (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

            pred = np.zeros((OUT_DIM,), dtype=np.uint8)
            pred[:4] = (prob[:4] >= THR_DIR).astype(np.uint8)
            pred[4:] = (prob[4:] >= THR_TAP).astype(np.uint8)

            # FPS
            t_now = time.time()
            fps = 1.0 / max(1e-6, (t_now - t_prev))
            t_prev = t_now

            # 박스/텍스트 표시
            px, py, pw, ph = pbox
            mx, my, mw, mh = mbox
            cv2.rectangle(frame, (px, py), (px+pw, py+ph), (0,255,0), 2)
            cv2.rectangle(frame, (mx, my), (mx+mw, my+mh), (0,0,255), 2)

            text = "PRED " + " ".join([f"{k}:{int(pred[i])}({prob[i]:.2f})" for i,k in enumerate(KEYS)])
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,0), 2)
            cv2.putText(frame, f"score P:{ps:.2f} M:{ms:.2f} | FPS:{fps:.1f} | q:quit",
                        (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)

            cv2.imshow("Live Preview (img+state policy -> overlay only)", frame)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
