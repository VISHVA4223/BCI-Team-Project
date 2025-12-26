import math
import di_input
import cv2
import mediapipe as mp

# --- CRITICAL: Restore these missing lines! ---
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Initialize MediaPipe Hands
hands = mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# Font for display
font = cv2.FONT_HERSHEY_SIMPLEX

# Webcam input
cap = cv2.VideoCapture(0)

# Configuration constants
CONFIDENCE_THRESHOLD = 0.5
RADIUS = 150
MIN_VERTICAL_DIFF = 65
WINDOW_NAME = 'MediaPipe Hands'
DARK_BLUE_COLOR = (102, 51, 0)

# --- NEW: Add variables for "Open Hands" detection ---
space_key_active = False
# For debouncing: only activate if open hands are detected for N consecutive frames
OPEN_HANDS_CONSECUTIVE_FRAMES = 3
consecutive_open_frames = 0

# Now use the 'hands' object in the 'with' statement
with hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

        # Store original image dimensions
        imageHeight, imageWidth, _ = image.shape

        # STEP 1: Convert to RGB for processing
        image.flags.writeable = False
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        # STEP 2: make image writeable again for drawing
        image.flags.writeable = True

        co = []  # List to store wrist coordinates
        open_hands_count = 0  # Count how many hands are detected as "open"

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks <-- This requires mp_drawing!
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

                # Extract WRIST landmark in pixel coordinates
                try:
                    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                    px, py = mp_drawing._normalized_to_pixel_coordinates(
                        wrist.x, wrist.y, imageWidth, imageHeight
                    )
                    if px is not None and py is not None:
                        co.append([px, py])
                except Exception:
                    continue

                # --- NEW: Check if this hand is "Open" using a more robust method ---
                try:
                    finger_tips = [
                        mp_hands.HandLandmark.THUMB_TIP,
                        mp_hands.HandLandmark.INDEX_FINGER_TIP,
                        mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                        mp_hands.HandLandmark.RING_FINGER_TIP,
                        mp_hands.HandLandmark.PINKY_TIP
                    ]
                    finger_bases = [
                        mp_hands.HandLandmark.THUMB_MCP,
                        mp_hands.HandLandmark.INDEX_FINGER_MCP,
                        mp_hands.HandLandmark.MIDDLE_FINGER_MCP,
                        mp_hands.HandLandmark.RING_FINGER_MCP,
                        mp_hands.HandLandmark.PINKY_MCP
                    ]

                    def get_pixel_coord(landmark):
                        return mp_drawing._normalized_to_pixel_coordinates(
                            landmark.x, landmark.y, imageWidth, imageHeight
                        )

                    total_dist_ratio = 0.0
                    valid_fingers = 0

                    for tip_id, base_id in zip(finger_tips, finger_bases):
                        tip = hand_landmarks.landmark[tip_id]
                        base = hand_landmarks.landmark[base_id]

                        tip_px, tip_py = get_pixel_coord(tip)
                        base_px, base_py = get_pixel_coord(base)

                        if None in (tip_px, tip_py, base_px, base_py):
                            continue

                        dist_base_tip = math.hypot(tip_px - base_px, tip_py - base_py)

                        pip_id = tip_id - 2
                        if pip_id < 0:
                            continue
                        pip = hand_landmarks.landmark[pip_id]
                        pip_px, pip_py = get_pixel_coord(pip)
                        if None in (pip_px, pip_py):
                            continue

                        dist_base_pip = math.hypot(pip_px - base_px, pip_py - base_py)

                        if dist_base_tip > dist_base_pip * 1.5:
                            total_dist_ratio += 1.0
                        else:
                            total_dist_ratio += 0.0

                        valid_fingers += 1

                    if valid_fingers > 0 and (total_dist_ratio / valid_fingers) > 0.8:
                        open_hands_count += 1

                except Exception as e:
                    print(f"Error detecting open hand: {e}")
                    continue

        # --- Two hands detected ---
        if len(co) == 2:
            x1, y1 = co[0]
            x2, y2 = co[1]

            xm = (x1 + x2) / 2
            ym = (y1 + y2) / 2

            # --- NEW: Check for "Open Hands" Pose with Debouncing ---
            if open_hands_count == 2:
                consecutive_open_frames += 1
                if consecutive_open_frames >= OPEN_HANDS_CONSECUTIVE_FRAMES:
                    if not space_key_active:
                        print("Space Key Activated! (Open Hands)")
                        di_input.press_key(di_input.DIK_SPACE)
                        space_key_active = True

                    left = min(x1, x2)
                    right = max(x1, x2)
                    top = min(y1, y2)
                    bottom = max(y1, y2)

                    padding = 30
                    left = max(0, int(left - padding))
                    right = min(imageWidth, int(right + padding))
                    top = max(0, int(top - padding))
                    bottom = min(imageHeight, int(bottom + padding))

                    cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 3)
                    cv2.putText(image, "SPACE ACTIVE", (left + 10, top + 30), font, 0.6,
                                (0, 0, 255), 2, cv2.LINE_AA)

            else:
                consecutive_open_frames = 0
                if space_key_active:
                    print("Space Key Released")
                    di_input.release_key(di_input.DIK_SPACE)
                    space_key_active = False

            # --- Existing Steering Logic (Unchanged) ---
            cv2.circle(image, center=(int(xm), int(ym)), radius=RADIUS,
                       color=DARK_BLUE_COLOR, thickness=15)

            dx = x2 - x1
            dy = y2 - y1

            action_taken = False

            if abs(dx) > 1e-6:
                m = dy / dx
            else:
                m = float('inf')

            def solve_perpendicular_intersections(m, xm, ym, radius):
                if m == float('inf'):
                    return [(xm, ym - radius), (xm, ym + radius)]
                elif m == 0:
                    return [(xm - radius, ym), (xm + radius, ym)]
                try:
                    m_perp = -1 / m
                    a = 1 + m_perp ** 2
                    c = xm**2 + ym**2 - radius**2
                    b = -2 * xm
                    discriminant = b**2 - 4*a*c
                    if discriminant < 0:
                        return [(xm, ym - radius), (xm, ym + radius)]

                    sqrt_disc = math.sqrt(discriminant)
                    xa = (-b + sqrt_disc) / (2 * a)
                    xb = (-b - sqrt_disc) / (2 * a)
                    ya = m_perp * (xa - xm) + ym
                    yb = m_perp * (xb - xm) + ym
                    return [(xa, ya), (xb, yb)]
                except:
                    return [(xm, ym - radius), (xm, ym + radius)]

            try:
                if m != float('inf'):
                    a_val = 1 + m**2
                    b_val = -2*xm - 2*m**2*x1 + 2*m*y1 - 2*m*ym
                    c_val = (xm**2 + m**2*x1**2 + y1**2 + ym**2 - 2*y1*ym - 2*m*y1*x1 + 2*m*ym*x1 - RADIUS**2)
                    disc = b_val**2 - 4*a_val*c_val
                    if disc >= 0:
                        sqrt_disc = math.sqrt(disc)
                        xa = (-b_val + sqrt_disc) / (2*a_val)
                        xb = (-b_val - sqrt_disc) / (2*a_val)
                        ya = m*(xa - x1) + y1
                        yb = m*(xb - x1) + y1
                    else:
                        xa, ya = xm - RADIUS, ym
                        xb, yb = xm + RADIUS, ym
                else:
                    xa = xb = xm
                    ya, yb = ym - RADIUS, ym + RADIUS

                cv2.line(image, (int(xa), int(ya)), (int(xb), int(yb)),
                         DARK_BLUE_COLOR, 20)

                perp_points = solve_perpendicular_intersections(m, xm, ym, RADIUS)
                xap, yap = perp_points[0]
                xbp, ybp = perp_points[1]

                if y1 > y2 and x1 > x2 and (y1 - y2) > MIN_VERTICAL_DIFF:
                    print("Turn left.")
                    di_input.release_key(di_input.DIK_S)
                    di_input.release_key(di_input.DIK_D)
                    di_input.press_key(di_input.DIK_A)
                    cv2.putText(image, "Turn left", (50, 50), font, 0.8,
                                (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.line(image, (int(xbp), int(ybp)), (int(xm), int(ym)),
                             DARK_BLUE_COLOR, 20)
                    action_taken = True

                elif y2 > y1 and x2 > x1 and (y2 - y1) > MIN_VERTICAL_DIFF:
                    print("Turn left.")
                    di_input.release_key(di_input.DIK_S)
                    di_input.release_key(di_input.DIK_D)
                    di_input.press_key(di_input.DIK_A)
                    cv2.putText(image, "Turn left", (50, 50), font, 0.8,
                                (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.line(image, (int(xbp), int(ybp)), (int(xm), int(ym)),
                             DARK_BLUE_COLOR, 20)
                    action_taken = True

                elif y2 > y1 and x1 > x2 and (y2 - y1) > MIN_VERTICAL_DIFF:
                    print("Turn right.")
                    di_input.release_key(di_input.DIK_S)
                    di_input.release_key(di_input.DIK_A)
                    di_input.press_key(di_input.DIK_D)
                    cv2.putText(image, "Turn right", (50, 50), font, 0.8,
                                (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.line(image, (int(xap), int(yap)), (int(xm), int(ym)),
                             DARK_BLUE_COLOR, 20)
                    action_taken = True

                elif y1 > y2 and x2 > x1 and (y1 - y2) > MIN_VERTICAL_DIFF:
                    print("Turn right.")
                    di_input.release_key(di_input.DIK_S)
                    di_input.release_key(di_input.DIK_A)
                    di_input.press_key(di_input.DIK_D)
                    cv2.putText(image, "Turn right", (50, 50), font, 0.8,
                                (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.line(image, (int(xap), int(yap)), (int(xm), int(ym)),
                             DARK_BLUE_COLOR, 20)
                    action_taken = True

                if not action_taken:
                    print("Keeping straight")
                    di_input.release_key(di_input.DIK_S)
                    di_input.release_key(di_input.DIK_A)
                    di_input.release_key(di_input.DIK_D)
                    di_input.press_key(di_input.DIK_W)
                    cv2.putText(image, "keep straight", (50, 50), font, 0.8,
                                (0, 255, 0), 2, cv2.LINE_AA)
                    if ybp > yap:
                        cv2.line(image, (int(xbp), int(ybp)), (int(xm), int(ym)),
                                 DARK_BLUE_COLOR, 20)
                    else:
                        cv2.line(image, (int(xap), int(yap)), (int(xm), int(ym)),
                                 DARK_BLUE_COLOR, 20)

            except Exception as e:
                print(f"Math error: {e}, going straight")
                di_input.release_key(di_input.DIK_S)
                di_input.release_key(di_input.DIK_A)
                di_input.release_key(di_input.DIK_D)
                di_input.press_key(di_input.DIK_W)
                cv2.putText(image, "STRAIGHT (ERR)", (50, 80), font, 0.7,
                            (0, 0, 255), 2, cv2.LINE_AA)

        # --- One hand detected ---
        elif len(co) == 1:
            print("Keeping back")
            di_input.release_key(di_input.DIK_A)
            di_input.release_key(di_input.DIK_D)
            di_input.release_key(di_input.DIK_W)
            di_input.press_key(di_input.DIK_S)
            cv2.putText(image, "keeping back", (50, 50), font, 1.0,
                        (0, 255, 0), 2, cv2.LINE_AA)

        else:
            di_input.release_key(di_input.DIK_W)
            di_input.release_key(di_input.DIK_A)
            di_input.release_key(di_input.DIK_S)
            di_input.release_key(di_input.DIK_D)
            if space_key_active:
                di_input.release_key(di_input.DIK_SPACE)
                space_key_active = False

        cv2.rectangle(image, (10, 10), (imageWidth - 10, imageHeight - 10),
                     DARK_BLUE_COLOR, 3)
        cv2.putText(image, "Virtual Steering Wheel", (imageWidth // 2 - 150, 30),
                   font, 0.8, DARK_BLUE_COLOR, 2, cv2.LINE_AA)

        cv2.imshow(WINDOW_NAME, cv2.flip(image, 1))

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
