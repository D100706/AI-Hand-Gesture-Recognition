import cv2
import mediapipe as mp
import math
import time

# ==================================================
# AI DUAL-HAND GESTURE RECOGNITION SYSTEM
# Supports 12 gestures and 2 hands simultaneously
# ==================================================

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles


# ==================================================
# DISTANCE CALCULATION
# ==================================================

def distance(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


# ==================================================
# FINGER DETECTION
# ==================================================

def classify_gesture(hand_landmarks):

    lm = hand_landmarks.landmark

    # Detect extended fingers
    index_up = lm[8].y < lm[6].y
    middle_up = lm[12].y < lm[10].y
    ring_up = lm[16].y < lm[14].y
    pinky_up = lm[20].y < lm[18].y

    # Thumb detection
    thumb_tip = lm[4]
    thumb_ip = lm[3]
    thumb_mcp = lm[2]

    thumb_up = (
        thumb_tip.y < thumb_ip.y
        and thumb_ip.y < thumb_mcp.y
    )

    # Thumb extension
    thumb_extended = distance(lm[4], lm[5]) > 0.12

    # Distance between thumb and index finger
    thumb_index_distance = distance(lm[4], lm[8])

    # ==================================================
    # GESTURE CLASSIFICATION
    # ==================================================

    # 1. OK SIGN
    if (
        thumb_index_distance < 0.045
        and middle_up and ring_up and pinky_up
    ):
        return "OK SIGN"

    # 2. PINCH
    if thumb_index_distance < 0.045:
        return "PINCH"

    # 3. THUMBS UP
    if (
        thumb_up
        and not index_up
        and not middle_up
        and not ring_up
        and not pinky_up
    ):
        return "THUMBS UP"

    # 4. OPEN HAND
    if (
        index_up and middle_up
        and ring_up and pinky_up
        and thumb_extended
    ):
        return "OPEN HAND"

    # 5. FIST
    if (
        not index_up and not middle_up
        and not ring_up and not pinky_up
    ):
        return "FIST"

    # 6. VICTORY / PEACE
    if (
        index_up and middle_up
        and not ring_up and not pinky_up
    ):
        return "VICTORY / PEACE"

    # 7. POINTING UP
    if (
        index_up and not middle_up
        and not ring_up and not pinky_up
    ):
        return "POINTING UP"

    # 8. THREE FINGERS
    if (
        index_up and middle_up
        and ring_up and not pinky_up
    ):
        return "THREE FINGERS"

    # 9. FOUR FINGERS
    if (
        index_up and middle_up
        and ring_up and pinky_up
        and not thumb_extended
    ):
        return "FOUR FINGERS"

    # 10. ROCK SIGN
    if (
        index_up and not middle_up
        and not ring_up and pinky_up
    ):
        return "ROCK SIGN"

    # 11. CALL ME
    if (
        not index_up and not middle_up
        and not ring_up and pinky_up
        and thumb_extended
    ):
        return "CALL ME"

    # 12. I LOVE YOU
    if (
        index_up and not middle_up
        and not ring_up and pinky_up
        and thumb_extended
    ):
        return "I LOVE YOU"

    return "GESTURE NOT RECOGNIZED"


# ==================================================
# DRAW DASHBOARD
# ==================================================

def draw_dashboard(frame, fps, gesture_count, hands_detected):

    h, w = frame.shape[:2]

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (10, 10),
        (w - 10, 150),
        (15, 20, 32),
        -1
    )

    cv2.addWeighted(
        overlay,
        0.82,
        frame,
        0.18,
        0,
        frame
    )

    cv2.putText(
        frame,
        "AI DUAL-HAND GESTURE RECOGNITION",
        (25, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"HANDS DETECTED: {hands_detected}/2",
        (25, 85),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (80, 220, 120),
        2
    )

    cv2.putText(
        frame,
        f"GESTURE CHANGES: {gesture_count}",
        (25, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 220, 90),
        2
    )

    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (w - 150, h - 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "PRESS Q TO QUIT",
        (25, h - 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (220, 220, 220),
        1
    )


# ==================================================
# DISPLAY HAND LABELS
# ==================================================

def draw_hand_label(frame, hand_landmarks, handedness, gesture):

    h, w = frame.shape[:2]

    wrist = hand_landmarks.landmark[0]

    x = int(wrist.x * w)
    y = int(wrist.y * h)

    label = f"{handedness}: {gesture}"

    text_x = max(10, min(x - 90, w - 300))
    text_y = max(175, min(y - 25, h - 40))

    cv2.putText(
        frame,
        label,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 120),
        2,
        cv2.LINE_AA
    )


# ==================================================
# MAIN APPLICATION
# ==================================================

def main():

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():

        print("ERROR: Could not open webcam.")
        print("Check camera permissions or try camera index 1.")

        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    previous_time = time.time()

    fps = 0.0

    gesture_count = 0

    previous_gestures = {}

    try:

        with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=0.65,
            min_tracking_confidence=0.60
        ) as hands:

            while True:

                success, frame = cap.read()

                if not success:

                    print("ERROR: Unable to read webcam.")
                    break

                # Mirror webcam
                frame = cv2.flip(frame, 1)

                # Convert image to RGB
                rgb = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

                rgb.flags.writeable = False

                results = hands.process(rgb)

                rgb.flags.writeable = True

                hands_detected = 0

                current_gestures = {}

                # ======================================
                # PROCESS BOTH HANDS
                # ======================================

                if results.multi_hand_landmarks:

                    hands_detected = len(
                        results.multi_hand_landmarks
                    )

                    for index, hand_landmarks in enumerate(
                        results.multi_hand_landmarks
                    ):

                        # Draw hand landmarks
                        mp_drawing.draw_landmarks(
                            frame,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_styles.get_default_hand_landmarks_style(),
                            mp_styles.get_default_hand_connections_style()
                        )

                        # Classify gesture
                        gesture = classify_gesture(
                            hand_landmarks
                        )

                        # Identify hand
                        handedness = (
                            results.multi_handedness[index]
                            .classification[0].label
                        )

                        # Store individual gesture
                        current_gestures[handedness] = gesture

                        # Draw gesture label
                        draw_hand_label(
                            frame,
                            hand_landmarks,
                            handedness,
                            gesture
                        )

                        # Count gesture changes separately
                        if (
                            gesture != "GESTURE NOT RECOGNIZED"
                            and gesture != previous_gestures.get(handedness)
                        ):

                            gesture_count += 1

                # Update previous gestures
                previous_gestures = current_gestures.copy()

                # ======================================
                # FPS CALCULATION
                # ======================================

                current_time = time.time()

                elapsed = current_time - previous_time

                if elapsed > 0:

                    instant_fps = 1.0 / elapsed

                    if fps == 0:

                        fps = instant_fps

                    else:

                        fps = (
                            0.9 * fps
                            + 0.1 * instant_fps
                        )

                previous_time = current_time

                # ======================================
                # DISPLAY DASHBOARD
                # ======================================

                draw_dashboard(
                    frame,
                    fps,
                    gesture_count,
                    hands_detected
                )

                cv2.imshow(
                    "AI Dual-Hand Gesture Recognition",
                    frame
                )

                # Press Q to exit
                if cv2.waitKey(1) & 0xFF == ord("q"):

                    break

    except Exception as exc:

        print(f"An error occurred: {exc}")

    finally:

        cap.release()

        cv2.destroyAllWindows()


# ==================================================
# START APPLICATION
# ==================================================

if __name__ == "__main__":

    main()

