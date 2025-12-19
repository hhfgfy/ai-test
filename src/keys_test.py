# keys_test.py
from pynput import keyboard
import time

pressed = set()

def on_press(key):
    pressed.add(key)

def on_release(key):
    if key in pressed:
        pressed.remove(key)

def get_action_vec():
    up = 1 if keyboard.Key.up in pressed else 0
    down = 1 if keyboard.Key.down in pressed else 0
    left = 1 if keyboard.Key.left in pressed else 0
    right = 1 if keyboard.Key.right in pressed else 0
    shift = 1 if (keyboard.Key.shift in pressed or keyboard.Key.shift_r in pressed) else 0
    return [up, down, left, right, shift]

listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

print("Press arrow keys / shift. Ctrl+C to quit.")
try:
    while True:
        print(get_action_vec())
        time.sleep(0.2)
except KeyboardInterrupt:
    pass
finally:
    listener.stop()
