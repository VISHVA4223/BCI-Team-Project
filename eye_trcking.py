import cv2
import mediapipe as mp
import pyautogui
import numpy as np

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Get screen dimensions
screen_width, screen_height = pyautogui.size()

# Configuration
CAM_WIDTH, CAM_HEIGHT = 640, 480
SMOOTHING_FACTOR = 0.75
EYE_OPEN_THRESHOLD = 0.25  # Adjust based on your eye shape (0.2 - 0.3)

# Initialize camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

# Smoothing buffers
prev_x = screen_width // 2
prev_y = screen_height // 2

# State variables for clicks
left_click_active = False
right_click_active = False

# Eye landmark indices for left and right eyes (MediaPipe)
LEFT_EYE_TOP_BOTTOM = [159, 145]   # Top & Bottom eyelid for left eye
RIGHT_EYE_TOP_BOTTOM = [386, 374]  # Top & Bottom eyelid for right eye

LEFT_EYE_INDICES = [33, 133, 160, 159, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [362, 263, 385, 384, 383, 263, 374, 380]

def calculate_eye_openness(landmarks, top_idx, bottom_idx):
    """Calculate vertical eye openness (normalized)"""
    top = landmarks[top_idx]
    bottom = landmarks[bottom_idx]
    return abs(top.y - bottom.y)

def is_eye_closed(landmarks, top_idx, bottom_idx, threshold=EYE_OPEN_THRESHOLD):
    """Check if eye is closed"""
    openness = calculate_eye_openness(landmarks, top_idx, bottom_idx)
    return openness < threshold

# Initialize Face Mesh
with mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:

    print("=== Eye Tracking with Blink-Based Clicks ===")
    print("Left Eye Closed → LEFT CLICK")
    print("Right Eye Closed → RIGHT CLICK")
    print("Press 'q' to quit")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        # Flip the frame horizontally for a mirror view
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Draw face landmarks (optional)
                mp_drawing.draw_landmarks(
                    frame,
                    face_landmarks,
                    mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )

                # Get left eye center
                left_eye_points = [(face_landmarks.landmark[i].x, face_landmarks.landmark[i].y) for i in LEFT_EYE_INDICES]
                left_eye_center = np.mean(left_eye_points, axis=0)
                left_eye_x = int(left_eye_center[0] * CAM_WIDTH)
                left_eye_y = int(left_eye_center[1] * CAM_HEIGHT)

                # Get right eye center
                right_eye_points = [(face_landmarks.landmark[i].x, face_landmarks.landmark[i].y) for i in RIGHT_EYE_INDICES]
                right_eye_center = np.mean(right_eye_points, axis=0)
                right_eye_x = int(right_eye_center[0] * CAM_WIDTH)
                right_eye_y = int(right_eye_center[1] * CAM_HEIGHT)

                # Draw red circles around eyes
                cv2.circle(frame, (left_eye_x, left_eye_y), 15, (0, 0, 255), 3)   # Red circle
                cv2.circle(frame, (right_eye_x, right_eye_y), 15, (0, 0, 255), 3) # Red circle

                # Calculate eye openness
                left_eye_open = calculate_eye_openness(face_landmarks.landmark, 159, 145)
                right_eye_open = calculate_eye_openness(face_landmarks.landmark, 386, 374)

                # Detect left eye closed → LEFT CLICK
                if is_eye_closed(face_landmarks.landmark, 159, 145):
                    if not left_click_active:
                        left_click_active = True
                        pyautogui.click(button='left')
                        print("LEFT CLICK triggered (Left Eye Closed)")
                else:
                    left_click_active = False

                # Detect right eye closed → RIGHT CLICK
                if is_eye_closed(face_landmarks.landmark, 386, 374):
                    if not right_click_active:
                        right_click_active = True
                        pyautogui.click(button='right')
                        print("RIGHT CLICK triggered (Right Eye Closed)")
                else:
                    right_click_active = False

                # Average eye position for cursor control
                avg_eye_x = (left_eye_center[0] + right_eye_center[0]) / 2
                avg_eye_y = (left_eye_center[1] + right_eye_center[1]) / 2

                # Map to screen coordinates
                cursor_x = int(avg_eye_x * screen_width)
                cursor_y = int(avg_eye_y * screen_height)

                # Apply smoothing
                prev_x = prev_x + SMOOTHING_FACTOR * (cursor_x - prev_x)
                prev_y = prev_y + SMOOTHING_FACTOR * (cursor_y - prev_y)

                # Move mouse cursor
                pyautogui.moveTo(int(prev_x), int(prev_y), duration=0, _pause=False)

                # Show status text
                status_text = "Eyes Open"
                if left_click_active:
                    status_text = "LEFT CLICK"
                elif right_click_active:
                    status_text = "RIGHT CLICK"

                cv2.putText(frame, status_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        else:
            cv2.putText(frame, "No face detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Display instructions
        cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0]-20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Show the frame
        cv2.imshow('Eye Tracking with Blink Clicks', frame)

        # Handle key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Cleanup
cap.release()
cv2.destroyAllWindows()
print("Eye Tracking with Blink-Based Clicks stopped")