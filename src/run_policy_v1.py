import os
import time
import cv2
import numpy as np
import mss
import torch
import torch.nn as nn
import torchvision.models as models
from collections import deque

import pyautogui

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.0

# =========================
# 모델/데이터 설정
# =========================
CKPT_PATH = "policy_img_state_stack4.pt"
IMG_SIZE = 96
STACK = 4
IN_CH = 3 * STACK
OUT_DIM = 6
STATE_DIM = 6
KEYS = ["UP", "DOWN", "LEFT", "RIGHT", "ALT", "SHIFT"]

# ---- 임계값 ----
THR_DIR = 0.20
THR_TAP = 0.20

# ---- 흔들림 완화 ----
EMA_ALPHA = 0.35

# ---- "HOLD 대신 연타" 주기(핵심) ----
# 0.03~0.06 추천: 30ms(빠름) ~ 60ms(느림)
REPEAT_INTERVAL_SEC = 0.04

# ---- 방향키 릴리즈 딜레이(OFF가 조금 흔들려도 계속 누르는 효과) ----
RELEASE_GRACE_SEC = 0.20

# ---- ALT/SHIFT tap ----
TAP_COOLDOWN = 0.12

PREVIEW_MAX_WIDTH = 1200

# ====== 템플릿 경로 ======
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PLAYER_TMPL = os.path.join(BASE_DIR, "..", "image", "player.png")
MONSTER_TMPL = os.path.join(BASE_DIR, "..", "image", "monster.png")
METHOD = cv2.TM_CCOEFF_NORMED

# pyautogui 키 이름
PYA_KEYS = {
    "UP": "up",
    "DOWN": "down",
    "LEFT": "left",
    "RIGHT": "right",
    "ALT": "alt",
    "SHIFT": "shift",
}

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

    pxn, pyn = px / W, py / H
    mxn, myn = mx / W, my / H
    dx, dy = (mx - px) / W, (my - py) / H
    state = np.array([pxn, pyn, mxn, myn, dx, dy], dtype=np.float32)
    return state, (ps, ms), pbox, mbox

def preprocess_to_tensor(frame_bgr):
    img = cv2.resize(frame_bgr, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    return x

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
class ImgStatePolicyStack(nn.Module):
    def __init__(self, out_dim=6, state_dim=6, in_ch=12):
        super().__init__()
        backbone = models.resnet18(weights=None)
        old = backbone.conv1
        backbone.conv1 = nn.Conv2d(
            in_ch, old.out_channels,
            kernel_size=old.kernel_size,
            stride=old.stride,
            padding=old.padding,
            bias=False
        )
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

# =========================
# 키 출력: "연타" 방식
# =========================
send_keys = False

# EMA 상태
prob_ema = None

# 표시용
current_prob_raw = np.zeros((OUT_DIM,), dtype=np.float32)
current_prob_filt = np.zeros((OUT_DIM,), dtype=np.float32)
current_active = set()

# 각 방향키별 반복 타이머
last_repeat_time = {k: 0.0 for k in ["UP", "DOWN", "LEFT", "RIGHT"]}
# OFF가 잠깐 나와도 바로 끊지 않기 위한 grace
dir_off_since = {k: None for k in ["UP", "DOWN", "LEFT", "RIGHT"]}
# ALT/SHIFT 탭 쿨다운
last_tap_time = {"ALT": 0.0, "SHIFT": 0.0}

def update_keys(prob_raw: np.ndarray, now: float):
    global prob_ema, current_prob_raw, current_prob_filt, current_active

    current_prob_raw = prob_raw.astype(np.float32)

    # EMA
    if prob_ema is None:
        prob_ema = prob_raw.astype(np.float32).copy()
    else:
        prob_ema = (1.0 - EMA_ALPHA) * prob_ema + EMA_ALPHA * prob_raw.astype(np.float32)

    prob = prob_ema
    current_prob_filt = prob.astype(np.float32)

    if not send_keys:
        current_active = set()
        # 연타 방식은 "다운 유지"가 없어서 별도 release 불필요
        return

    # 방향키 want 계산 + grace
    want_dir = {}
    for i, k in enumerate(["UP", "DOWN", "LEFT", "RIGHT"]):
        p = float(prob[i])
        want = (p >= THR_DIR)
        if want:
            dir_off_since[k] = None
        else:
            if dir_off_since[k] is None:
                dir_off_since[k] = now
            # OFF가 grace 미만이면 "계속 누르는 것으로 간주"
            if (now - dir_off_since[k]) < RELEASE_GRACE_SEC:
                want = True
        want_dir[k] = want

    # 활성 키 표시용
    current_active = {k for k,v in want_dir.items() if v}
    if float(prob[4]) >= THR_TAP: current_active.add("ALT")
    if float(prob[5]) >= THR_TAP: current_active.add("SHIFT")

    # 방향키 연타(반복 press)
    for k, want in want_dir.items():
        if not want:
            continue
        if (now - last_repeat_time[k]) >= REPEAT_INTERVAL_SEC:
            pyautogui.press(PYA_KEYS[k])
            last_repeat_time[k] = now

    # ALT/SHIFT tap
    for idx, k in [(4, "ALT"), (5, "SHIFT")]:
        if float(prob[idx]) >= THR_TAP and (now - last_tap_time[k] >= TAP_COOLDOWN):
            pyautogui.press(PYA_KEYS[k])
            last_tap_time[k] = now

def overlay(frame: np.ndarray, fps: float):
    x0, y0 = 10, 20
    onoff = "ON" if send_keys else "OFF"
    active = " ".join(sorted(list(current_active))) if current_active else "(none)"

    cv2.putText(frame, f"FPS:{fps:.1f} | q:quit | s:toggle key output",
                (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)
    cv2.putText(frame, f"KEY OUTPUT: {onoff} (repeat-press mode)",
                (x0, y0+25), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0,200,255), 2)
    cv2.putText(frame, f"ACTIVE: {active}",
                (x0, y0+55), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,255), 2)

    raw_str = " ".join([f"{KEYS[i]}={float(current_prob_raw[i]):.2f}" for i in range(OUT_DIM)])
    filt_str = " ".join([f"{KEYS[i]}={float(current_prob_filt[i]):.2f}" for i in range(OUT_DIM)])
    cv2.putText(frame, f"raw : {raw_str}",
                (x0, y0+80), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1)
    cv2.putText(frame, f"filt: {filt_str}",
                (x0, y0+100), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1)

    cv2.putText(frame,
                f"THR_DIR={THR_DIR:.2f} REPEAT={REPEAT_INTERVAL_SEC*1000:.0f}ms GRACE={RELEASE_GRACE_SEC:.2f}s EMA={EMA_ALPHA:.2f}",
                (x0, y0+125), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1)

def main():
    global send_keys

    player_t = cv2.imread(PLAYER_TMPL)
    monster_t = cv2.imread(MONSTER_TMPL)
    if player_t is None or monster_t is None:
        print("Template load failed.")
        print("PLAYER_TMPL:", PLAYER_TMPL)
        print("MONSTER_TMPL:", MONSTER_TMPL)
        return

    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    model = ImgStatePolicyStack(out_dim=OUT_DIM, state_dim=STATE_DIM, in_ch=IN_CH)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    region = pick_roi_from_screen(monitor_index=1, preview_max_width=PREVIEW_MAX_WIDTH)
    if region is None:
        print("ROI selection canceled.")
        return

    buf = deque(maxlen=STACK)

    WIN_TITLE = "Live Preview (policy -> pyautogui repeat-press)"
    cv2.namedWindow(WIN_TITLE, cv2.WINDOW_NORMAL)

    print("[INFO] q: quit, s: toggle key output")
    print("[INFO] ON 하면 3초 안에 대상 창(게임/메모장)을 클릭해서 포커스 유지하세요.")
    print("[INFO] pyautogui FAILSAFE=True: 마우스를 화면 모서리로 보내면 즉시 중단됩니다.")

    t_prev = time.time()

    try:
        with mss.mss() as sct:
            while True:
                img = np.array(sct.grab(region))
                frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

                state_np, (ps, ms), pbox, mbox = make_state_from_frame(frame, player_t, monster_t)

                x3 = preprocess_to_tensor(frame)
                buf.append(x3)
                if len(buf) < STACK:
                    while len(buf) < STACK:
                        buf.appendleft(buf[0])

                x_img = torch.cat(list(buf), dim=0).unsqueeze(0)
                x_state = torch.from_numpy(state_np).unsqueeze(0)

                with torch.no_grad():
                    prob_raw = torch.sigmoid(model(x_img, x_state))[0].numpy()

                t_now = time.time()
                fps = 1.0 / max(1e-6, (t_now - t_prev))
                t_prev = t_now

                update_keys(prob_raw, t_now)

                # 박스 표시
                px, py, pw, ph = pbox
                mx, my, mw, mh = mbox
                cv2.rectangle(frame, (px, py), (px + pw, py + ph), (0,255,0), 2)
                cv2.rectangle(frame, (mx, my), (mx + mw, my + mh), (0,0,255), 2)

                overlay(frame, fps)
                cv2.imshow(WIN_TITLE, frame)

                k = cv2.waitKey(1) & 0xFF
                if k == ord("q"):
                    break

                if k == ord("s"):
                    if not send_keys:
                        print("[INFO] Key output ON in 3 seconds. Focus target window now...")
                        for i in range(3, 0, -1):
                            print(f"  {i}...")
                            time.sleep(1)
                        send_keys = True
                        print("[INFO] Key output is ON (repeat-press).")
                    else:
                        send_keys = False
                        print("[INFO] Key output is OFF.")

    finally:
        send_keys = False
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
