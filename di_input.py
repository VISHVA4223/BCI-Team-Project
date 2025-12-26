# di_input.py
# DirectInput Engine for Game-Compatible Key Simulation
# Uses Hardware Scan Codes (Works with Asphalt, NFS, GTA, etc.)

import ctypes

# --- SCAN CODES (QWERTY Layout) ---
# These are the actual hardware codes sent by a physical keyboard.
DIK_W = 0x11
DIK_A = 0x1E
DIK_S = 0x1F
DIK_D = 0x20
DIK_SPACE = 0x39  # This is the correct scan code for Space Bar
DIK_LSHIFT = 0x2A

# --- Windows API Structures ---
PUL = ctypes.POINTER(ctypes.c_ulong)

class KeyBdInput(ctypes.Structure):
    _fields_ = [("wVk", ctypes.c_ushort),
                ("wScan", ctypes.c_ushort),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]

class HardwareInput(ctypes.Structure):
    _fields_ = [("uMsg", ctypes.c_ulong),
                ("wParamL", ctypes.c_short),
                ("wParamH", ctypes.c_ushort)]

class MouseInput(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long),
                ("dy", ctypes.c_long),
                ("mouseData", ctypes.c_ulong),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]

class Input_I(ctypes.Union):
    _fields_ = [("ki", KeyBdInput),
                 ("mi", MouseInput),
                 ("hi", HardwareInput)]

class Input(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong),
                ("ii", Input_I)]

# --- Core Functions ---
def press_key(scan_code):
    """
    Presses a key using its hardware scan code.
    :param scan_code: The DIK_* constant (e.g., DIK_SPACE).
    """
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    # Use 0 for wVk (virtual key) and the scan code for wScan.
    # The 0x0008 flag is KEYEVENTF_SCANCODE.
    ii_.ki = KeyBdInput(0, scan_code, 0x0008, 0, ctypes.pointer(extra))
    x = Input(ctypes.c_ulong(1), ii_)
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

def release_key(scan_code):
    """
    Releases a key using its hardware scan code.
    :param scan_code: The DIK_* constant (e.g., DIK_SPACE).
    """
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    # 0x0008 | 0x0002 = KEYEVENTF_SCANCODE | KEYEVENTF_KEYUP
    ii_.ki = KeyBdInput(0, scan_code, 0x0008 | 0x0002, 0, ctypes.pointer(extra))
    x = Input(ctypes.c_ulong(1), ii_)
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))
