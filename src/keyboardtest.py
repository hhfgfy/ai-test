import time
import ctypes
from ctypes import wintypes

user32 = ctypes.WinDLL("user32", use_last_error=True)

# --- constants ---
INPUT_MOUSE = 0
INPUT_KEYBOARD = 1
INPUT_HARDWARE = 2

KEYEVENTF_EXTENDEDKEY = 0x0001
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_SCANCODE = 0x0008

VK_RIGHT = 0x27  # RIGHT arrow
MAPVK_VK_TO_VSC = 0

# ULONG_PTR 호환
ULONG_PTR = getattr(wintypes, "ULONG_PTR", ctypes.c_size_t)

# --- structures (Windows 공식 레이아웃과 동일하게 맞춤) ---
class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    ]

class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    ]

class HARDWAREINPUT(ctypes.Structure):
    _fields_ = [
        ("uMsg", wintypes.DWORD),
        ("wParamL", wintypes.WORD),
        ("wParamH", wintypes.WORD),
    ]

class _INPUTUNION(ctypes.Union):
    _fields_ = [
        ("mi", MOUSEINPUT),
        ("ki", KEYBDINPUT),
        ("hi", HARDWAREINPUT),
    ]

class INPUT(ctypes.Structure):
    _anonymous_ = ("u",)
    _fields_ = [
        ("type", wintypes.DWORD),
        ("u", _INPUTUNION),
    ]

# API signatures
user32.SendInput.argtypes = (wintypes.UINT, ctypes.POINTER(INPUT), ctypes.c_int)
user32.SendInput.restype = wintypes.UINT
user32.MapVirtualKeyW.argtypes = (wintypes.UINT, wintypes.UINT)
user32.MapVirtualKeyW.restype = wintypes.UINT

def _scancode(vk: int) -> int:
    return int(user32.MapVirtualKeyW(vk, MAPVK_VK_TO_VSC))

def send_scan_down(vk: int):
    # 방향키는 EXTENDEDKEY 플래그가 필요할 수 있음
    flags = KEYEVENTF_SCANCODE | KEYEVENTF_EXTENDEDKEY
    sc = _scancode(vk)

    inp = INPUT()
    inp.type = INPUT_KEYBOARD
    inp.ki = KEYBDINPUT(
        wVk=0,
        wScan=sc,
        dwFlags=flags,
        time=0,
        dwExtraInfo=ULONG_PTR(0),
    )

    sent = user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))
    if sent != 1:
        raise ctypes.WinError(ctypes.get_last_error())

def send_scan_up(vk: int):
    flags = KEYEVENTF_SCANCODE | KEYEVENTF_EXTENDEDKEY | KEYEVENTF_KEYUP
    sc = _scancode(vk)

    inp = INPUT()
    inp.type = INPUT_KEYBOARD
    inp.ki = KEYBDINPUT(
        wVk=0,
        wScan=sc,
        dwFlags=flags,
        time=0,
        dwExtraInfo=ULONG_PTR(0),
    )

    sent = user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))
    if sent != 1:
        raise ctypes.WinError(ctypes.get_last_error())

def main():
    print("=== SendInput RIGHT 테스트(정석 구조체) ===")
    print("5초 안에 메모장을 열고 커서 놓고, 메모장 창을 클릭해서 포그라운드로 만들어두세요.")
    for i in range(5, 0, -1):
        print(f"{i}...")
        time.sleep(1)

    print("RIGHT 누르는 중(1.0초)...")
    send_scan_down(VK_RIGHT)
    time.sleep(1.0)
    send_scan_up(VK_RIGHT)
    print("완료. 메모장 커서가 오른쪽으로 이동했나요?")

if __name__ == "__main__":
    main()
