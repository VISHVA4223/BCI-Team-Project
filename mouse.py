import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
from collections import deque
import math

# ================== CRITICAL FIXES ==================
# Disable PyAutoGUI fail-safe (safe for this controlled app)
pyautogui.FAILSAFE = False
# Add corner protection (15px buffer from edges)
CORNER_BUFFER = 15
# Minimum movement threshold to prevent micro-movements
MIN_MOVEMENT = 0.5
# ==================================================

# ================== PRECISION-FOCUSED CONFIGURATION ==================
# Gaming mode parameters (optimized for Fruit Ninja)
GAMING_MODE = True  # Set to False for normal mode
CURSOR_SPEED = 3.8        # For gaming: 3.5-4.5 (balanced for precision)
SMOOTHING_FACTOR = 0.25   # For gaming: 0.2-0.4 (lower = more responsive)
DEAD_ZONE = 0.0           # For gaming: 0.0 (normal: 0.003-0.007)
ACCELERATION_FACTOR = 1.7  # For gaming: 1.5-2.0 (normal: 1.2-1.5)
CLICK_DELAY = 0.04        # Reduced for rapid clicking
MAX_BUFFER_SIZE = 8       # For prediction and smoothing

# Normal mode parameters (for comparison)
NORMAL_SPEED = 2.0
NORMAL_SMOOTHING = 0.6
NORMAL_DEAD_ZONE = 0.005

# Gesture detection
LEFT_PINCH_THRESHOLD = 0.035
RIGHT_PINCH_THRESHOLD = 0.04
OPEN_HAND_THRESHOLD = 0.07
FIST_THRESHOLD = 0.06     # For gaming mode toggle

# Performance
TARGET_FPS = 120           # Higher FPS for gaming
# ==================================================

# Initialize
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
screen_width, screen_height = pyautogui.size()

# Camera
cam_width, cam_height = 640, 480
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_height)

# State management
clicking_left = False
clicking_right = False
gaming_mode = GAMING_MODE
prev_time = time.time()
frame_count = 0
last_click_time = 0
last_gesture_time = 0

# Initialize cursor at screen center (safe position)
cursor_x, cursor_y = screen_width // 2, screen_height // 2
prev_x, prev_y = cursor_x, cursor_y

# For advanced smoothing and prediction
position_history = deque(maxlen=MAX_BUFFER_SIZE)
velocity_history = deque(maxlen=MAX_BUFFER_SIZE)
confidence = 1.0  # Track confidence in hand tracking

def get_landmark(hand_landmarks, idx):
    """Get normalized landmark coordinates"""
    lm = hand_landmarks.landmark[idx]
    return np.array([lm.x, lm.y])

def is_thumb_index_pinch(hand_landmarks, threshold=LEFT_PINCH_THRESHOLD):
    """Check if thumb and index fingers are pinching"""
    thumb = get_landmark(hand_landmarks, 4)
    index = get_landmark(hand_landmarks, 8)
    return np.linalg.norm(thumb - index) < threshold

def is_thumb_pinky_pinch(hand_landmarks, threshold=RIGHT_PINCH_THRESHOLD):
    """Check if thumb and pinky fingers are pinching"""
    thumb = get_landmark(hand_landmarks, 4)
    pinky = get_landmark(hand_landmarks, 20)
    return np.linalg.norm(thumb - pinky) < threshold

def is_hand_open(hand_landmarks, threshold=OPEN_HAND_THRESHOLD):
    """Check if hand is fully open"""
    wrist = get_landmark(hand_landmarks, 0)
    tips = [4, 8, 12, 16, 20]
    avg_dist = np.mean([
        np.linalg.norm(wrist - get_landmark(hand_landmarks, i))
        for i in tips
    ])
    return avg_dist > threshold

def is_fist(hand_landmarks, threshold=FIST_THRESHOLD):
    """Check if hand is in fist (for gaming mode toggle)"""
    wrist = get_landmark(hand_landmarks, 0)
    tips = [4, 8, 12, 16, 20]
    avg_dist = np.mean([
        np.linalg.norm(wrist - get_landmark(hand_landmarks, i))
        for i in tips
    ])
    return avg_dist < threshold

def calculate_confidence(index_tip):
    """Calculate confidence based on index finger position and stability"""
    # Higher confidence when hand is centered and stable
    center_x = abs(0.5 - index_tip[0])
    center_y = abs(0.5 - index_tip[1])
    distance_from_center = np.sqrt(center_x**2 + center_y**2)
    
    # Confidence decreases as hand moves away from center
    confidence = max(0.3, 1.0 - distance_from_center * 1.5)
    
    # Add some randomness for smoothing
    confidence = confidence * 0.9 + np.random.uniform(0, 0.1)
    
    return confidence

def apply_advanced_smoothing(new_x, new_y):
    """Advanced smoothing with prediction for gaming"""
    global position_history, velocity_history, confidence
    
    # Add new position to history
    position_history.append((new_x, new_y))
    
    # Calculate velocity if we have enough history
    if len(position_history) >= 4:
        # Calculate current velocity
        dx = new_x - position_history[-2][0]
        dy = new_y - position_history[-2][1]
        velocity = (dx, dy)
        velocity_history.append(velocity)
        
        # Calculate average velocity
        avg_vx = np.mean([v[0] for v in velocity_history])
        avg_vy = np.mean([v[1] for v in velocity_history])
        
        # Predict next position based on velocity
        prediction_factor = 0.3 * confidence  # More prediction when confident
        predicted_x = new_x + avg_vx * prediction_factor
        predicted_y = new_y + avg_vy * prediction_factor
        
        # Weighted average with more weight on recent positions
        weights = np.linspace(0.3, 1.0, len(position_history))
        weighted_x = sum(x * w for (x, _) , w in zip(position_history, weights)) / sum(weights)
        weighted_y = sum(y * w for (_, y) , w in zip(position_history, weights)) / sum(weights)
        
        # Combine weighted average with prediction
        final_x = weighted_x * (1 - prediction_factor) + predicted_x * prediction_factor
        final_y = weighted_y * (1 - prediction_factor) + predicted_y * prediction_factor
        
        return final_x, final_y
    
    return new_x, new_y

def apply_adaptive_acceleration(dx, dy, speed):
    """Adaptive acceleration based on movement speed and direction"""
    # More aggressive acceleration for fast movements
    if speed > 0.05:
        acceleration = 1.0 + (ACCELERATION_FACTOR - 1.0) * min(1.0, speed / 0.03)
    else:
        # Smoother acceleration for slow movements
        acceleration = 1.0 + (ACCELERATION_FACTOR - 1.0) * (speed / 0.03) * 0.7
    
    return dx * acceleration, dy * acceleration

def toggle_gaming_mode():
    """Toggle between gaming and normal modes"""
    global gaming_mode, CURSOR_SPEED, SMOOTHING_FACTOR, DEAD_ZONE
    global ACCELERATION_FACTOR, MAX_BUFFER_SIZE
    
    gaming_mode = not gaming_mode
    
    if gaming_mode:
        CURSOR_SPEED = 3.8
        SMOOTHING_FACTOR = 0.25
        DEAD_ZONE = 0.0
        ACCELERATION_FACTOR = 1.7
        MAX_BUFFER_SIZE = 8
        print("GAMING MODE ACTIVATED")
    else:
        CURSOR_SPEED = NORMAL_SPEED
        SMOOTHING_FACTOR = NORMAL_SMOOTHING
        DEAD_ZONE = NORMAL_DEAD_ZONE
        ACCELERATION_FACTOR = 1.3
        MAX_BUFFER_SIZE = 12
        print("NORMAL MODE ACTIVATED")

# Performance monitoring
start_time = time.time()

with mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
) as hands:

    print(f"Ultra-Precise Gaming Mouse Controller | Target FPS: {TARGET_FPS}")
    print("Press 'q' to exit | 'g' to toggle gaming mode | 'r' to reset cursor")
    
    while cap.isOpened():
        current_time = time.time()
        frame_count += 1
        
        # Calculate actual FPS
        if current_time - start_time >= 1.0:
            actual_fps = frame_count / (current_time - start_time)
            print(f"FPS: {actual_fps:.1f} | Mode: {'GAMING' if gaming_mode else 'NORMAL'} | Confidence: {confidence:.2f}")
            frame_count = 0
            start_time = current_time
        
        # Process frame
        success, frame = cap.read()
        if not success:
            continue
        
        # Flip for mirror effect
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = hands.process(rgb_frame)
        
        # Process hand landmarks if detected
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            index_tip = get_landmark(hand_landmarks, 8)
            
            # Calculate confidence
            confidence = calculate_confidence(index_tip)
            
            # Convert normalized coordinates to screen coordinates
            raw_x = index_tip[0] * screen_width
            raw_y = index_tip[1] * screen_height
            
            # Calculate movement
            dx = raw_x - cursor_x
            dy = raw_y - cursor_y
            
            # Apply dead zone (0 for gaming)
            if DEAD_ZONE > 0:
                distance = np.sqrt(dx*dx + dy*dy)
                if distance < DEAD_ZONE:
                    dx, dy = 0, 0
                else:
                    scale = (distance - DEAD_ZONE) / distance
                    dx *= scale
                    dy *= scale
            
            # Apply adaptive acceleration
            speed = np.sqrt(dx*dx + dy*dy)
            dx, dy = apply_adaptive_acceleration(dx, dy, speed)
            
            # Apply sensitivity
            dx *= CURSOR_SPEED
            dy *= CURSOR_SPEED
            
            # Calculate new position
            new_x = cursor_x + dx
            new_y = cursor_y + dy
            
            # Apply advanced smoothing
            smooth_x, smooth_y = apply_advanced_smoothing(new_x, new_y)
            
            # === CRITICAL FIX: Corner Protection ===
            # Prevent cursor from reaching screen edges
            cursor_x = max(CORNER_BUFFER, min(screen_width - CORNER_BUFFER - 1, smooth_x))
            cursor_y = max(CORNER_BUFFER, min(screen_height - CORNER_BUFFER - 1, smooth_y))
            
            # Move mouse
            pyautogui.moveTo(cursor_x, cursor_y, duration=0, _pause=False)
            
            # === GESTURE DETECTION ===
            left_gesture = is_thumb_index_pinch(hand_landmarks)
            right_gesture = is_thumb_pinky_pinch(hand_landmarks)
            open_gesture = is_hand_open(hand_landmarks)
            fist_gesture = is_fist(hand_landmarks)
            
            # Toggle gaming mode with fist
            if fist_gesture and time.time() - last_gesture_time > 1.0:
                toggle_gaming_mode()
                last_gesture_time = time.time()
            
            # LEFT CLICK (for slicing)
            if left_gesture:
                current_time = time.time()
                if not clicking_left or (current_time - last_click_time > CLICK_DELAY):
                    pyautogui.mouseDown(button='left')
                    clicking_left = True
                    last_click_time = current_time
                cv2.putText(frame, "SLICING", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.circle(frame, (int(index_tip[0] * cam_width), int(index_tip[1] * cam_height)), 15, (0, 0, 255), 2)
            else:
                if clicking_left:
                    pyautogui.mouseUp(button='left')
                    clicking_left = False
            
            # Draw hand landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        else:
            # No hand detected → release left click
            if clicking_left:
                pyautogui.mouseUp(button='left')
                clicking_left = False
            
            # Show "no hand" feedback
            cv2.putText(frame, "NO HAND DETECTED", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Display mode
        mode_text = "GAMING MODE" if gaming_mode else "NORMAL MODE"
        cv2.putText(frame, mode_text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 
                   (0, 0, 255) if gaming_mode else (0, 255, 0), 2)
        
        # Confidence indicator
        cv2.putText(frame, f"Confidence: {confidence:.2f}", (10, 180), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Instructions
        cv2.putText(frame, "Move: Index Finger", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "Slice: Thumb+Index", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "Toggle Mode: Make Fist", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display frame
        cv2.imshow("Ultra-Precise Gaming Mouse Controller", frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('g'):
            toggle_gaming_mode()
        elif key == ord('r'):
            # Reset cursor to center
            cursor_x, cursor_y = screen_width // 2, screen_height // 2
            pyautogui.moveTo(cursor_x, cursor_y)
            position_history.clear()
            velocity_history.clear()
            print("Cursor position reset")

# Cleanup
cap.release()
cv2.destroyAllWindows()
print("Ultra-Precise Gaming Mouse Controller stopped")