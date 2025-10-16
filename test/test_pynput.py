# import pynput, os
# print("DISPLAY", os.environ.get("DISPLAY"))
# from pynput import keyboard
# print("pynput import ok")

from Xlib import display
d = display.Display()
print("X open:", bool(d))
for ext in ("RECORD","XInputExtension","XTEST"):
    try:
        print(ext, ":", d.has_extension(ext))
    except Exception as e:
        print(ext, "check error:", e)
