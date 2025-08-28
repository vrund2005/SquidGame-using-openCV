import cv2
import mediapipe as mp
import numpy as np
import time
import random

# Initialize mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


def distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def overlay_image(background, overlay, x, y):
    h, w = overlay.shape[:2]
    if overlay.shape[2] == 4:  # If PNG with alpha channel
        alpha = overlay[:, :, 3] / 255.0
        for c in range(3):
            background[y:y+h, x:x+w, c] = (alpha * overlay[:, :, c] +
                                           (1 - alpha) * background[y:y+h, x:x+w, c])
    else:
        background[y:y+h, x:x+w] = overlay
    return background


cap = cv2.VideoCapture(0)

# Load doll images (ensure PNG with transparent background)
doll_red = cv2.imread("/Users/vrund/Desktop/Sem_7/OpenCV/Squid_CV/doll_red.png", cv2.IMREAD_UNCHANGED)
doll_green = cv2.imread("/Users/vrund/Desktop/Sem_7/OpenCV/Squid_CV/doll_green.png", cv2.IMREAD_UNCHANGED)

if doll_red is None or doll_green is None:
    raise FileNotFoundError("One or both doll images failed to load. Check paths & formats.")


doll_red = cv2.resize(doll_red, (500,600))
doll_green = cv2.resize(doll_green, (500, 500))

# Game variables
game_duration = 60  # 30 seconds
game_start = time.time()
light = "Green Light"
next_switch = time.time() + random.randint(3, 7)
dead = False

# Red light timing
red_light_start = None
grace_period = 1  # seconds

# Movement detection variables
prev_positions = None
movement_threshold = 0.02  # tweak if needed

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip for mirror effect
        frame = cv2.flip(frame, 1)

        # Convert to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)

        # Convert back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        movement_detected = False

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Track nose and ankles
            tracked_points = {
                "nose": [landmarks[mp_pose.PoseLandmark.NOSE.value].x,
                         landmarks[mp_pose.PoseLandmark.NOSE.value].y],
                "l_ankle": [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y],
                "r_ankle": [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                            landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
            }

            if prev_positions is not None:
                distances = []
                for key in tracked_points:
                    dist = distance(tracked_points[key], prev_positions[key])
                    distances.append(dist)

                if max(distances) > movement_threshold:
                    movement_detected = True

            prev_positions = tracked_points

        # Switch lights at random intervals
        if time.time() > next_switch:
            light = "Red Light" if light == "Green Light" else "Green Light"
            next_switch = time.time() + random.randint(3, 7)

            if light == "Red Light":
                red_light_start = time.time()  # mark when red starts

        # Death condition with grace period
        if light == "Red Light":
            if red_light_start is not None and (time.time() - red_light_start) > grace_period:
                if movement_detected:
                    dead = True

        # Show game status
        if dead:
            cv2.putText(image, "YOU'RE DEAD", (650, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 7)
        elif time.time() - game_start >= game_duration:
            cv2.putText(image, "YOU WIN!", (650, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 7)
        else:
            cv2.putText(image, light, (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 3,
                        (0, 0, 255) if light == "Red Light" else (0, 255, 0), 5)


        # Add doll (top-right corner)
        if light == "Red Light":
            image = overlay_image(image, doll_red, 0, 450)
        else:
            image = overlay_image(image, doll_green, 0, 550)

        # Draw landmarks
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Show
        cv2.imshow("Red Light Green Light", image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        if dead or (time.time() - game_start >= game_duration):
            time.sleep(3)
            break

    cap.release()
    cv2.destroyAllWindows()