import time
from pynput import keyboard
import pyautogui

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.0

# =========================
# 기록 버퍼
# =========================
events = []  # (dt, type, key)
t_start = None
recording = True

print("===================================")
print(" 키보드 입력 기록 시작")
print(" ESC 를 누르면 기록 종료")
print("===================================")


def normalize_key(key):
    """pynput key -> pyautogui key string"""
    if isinstance(key, keyboard.Key):
        return key.name  # 'up', 'shift', 'esc' ...
    else:
        return key.char  # 'a', '1', ...


def on_press(key):
    global t_start, recording

    now = time.time()
    if t_start is None:
        t_start = now

    # ESC → 기록 종료
    if key == keyboard.Key.esc:
        recording = False
        print("[STOP] ESC pressed, stop recording")
        return False

    try:
        k = normalize_key(key)
        events.append((now - t_start, "down", k))
        print(f"DOWN  {k}")
    except Exception:
        pass


def on_release(key):
    if not recording:
        return

    now = time.time()
    try:
        k = normalize_key(key)
        events.append((now - t_start, "up", k))
        print(f"UP    {k}")
    except Exception:
        pass


# =========================
# 기록 단계
# =========================
with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
    listener.join()

print(f"\n[INFO] Recorded {len(events)} events")

if len(events) == 0:
    print("No events recorded. Exit.")
    exit()

# =========================
# 재생 대기
# =========================
print("\n===================================")
print(" 3초 후 입력 재생 시작")
print(" 재생할 창(게임/메모장 등)을 클릭해서 포커스 유지")
print("===================================")
for i in range(3, 0, -1):
    print(f"{i}...")
    time.sleep(1)

print("[PLAY] Replaying inputs...\n")

# =========================
# 재생 단계
# =========================
t0 = events[0][0]

for i, (t, typ, k) in enumerate(events):
    if i > 0:
        dt = t - events[i - 1][0]
        if dt > 0:
            time.sleep(dt)

    try:
        if typ == "down":
            pyautogui.keyDown(k)
        else:
            pyautogui.keyUp(k)
    except Exception as e:
        print("Skip key:", k, e)

print("\n[DONE] Replay finished")
