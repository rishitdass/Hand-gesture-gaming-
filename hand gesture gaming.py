import cv2
import mediapipe as mp
import numpy as np
import control
import time
id_hex=  {  3: 0x1F,#S
            1:0x11,#W
            4:0x20,#D
            2:0x1E}#A


# Initialize MediaPipe Hands with max_num_hands set to 2.
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.4, min_tracking_confidence=0.3)

# Example of predefined gesture coordinates (as you have defined earlier)
# A
reference_gesture_1 = {
0:(203,310),
1:(223,289),
2:(220,260),
3:(216,243),
4:(213,228),
5:(199,225),
6:(202,192),
7:(204,168),
8:(204,147),
9:(196,224),
10:(199,185),
11:(201,158),
12:(201,135),
13:(195,225),
14:(199,186),
15:(200,160),
16:(200,139),
17:(198,229),
18:(201,198),
19:(201,177),
20:(200,158)
}
# D
reference_gesture_2 = {
0:(159,330),
1:(197,345),
2:(234,338),
3:(259,335),
4:(280,334),
5:(248,310),
6:(288,319),
7:(316,322),
8:(339,323),
9:(245,309),
10:(294,319),
11:(326,323),
12:(353,322),
13:(240,316),
14:(288,324),
15:(316,326),
16:(340,324),
17:(233,328),
18:(268,332),
19:(291,331),
20:(311,330)
}
# S
reference_gesture_3 = {
0:(522,397),
1:(483,384),
2:(450,354),
3:(441,320),
4:(460,304),
5:(473,316),
6:(471,291),
7:(474,321),
8:(475,332),
9:(497,314),
10:(493,292),
11:(495,327),
12:(496,334),
13:(520,315),
14:(518,293),
15:(516,327),
16:(516,336),
17:(544,320),
18:(540,301),
19:(536,324),
20:(535,333)
}

# W
reference_gesture_4 = {
0:(473,372),
1:(432,354),
2:(403,319),
3:(388,286),
4:(372,261),
5:(438,258),
6:(429,214),
7:(425,187),
8:(423,163),
9:(464,252),
10:(462,201),
11:(461,169),
12:(460,142),
13:(487,257),
14:(490,211),
15:(491,181),
16:(492,155),
17:(509,270),
18:(518,237),
19:(523,214),
20:(527,193)
}


def is_gesture_match(landmarks, predefined_gesture, image_shape, distance_threshold=90, angle_threshold=70):
    h, w, _ = image_shape  # Get image dimensions

    # Convert landmarks to pixel positions
    landmark_positions = np.array([[lm.x * w, lm.y * h] for lm in landmarks])

    # Calculate the center of the reference gesture
    ref_positions = np.array(list(predefined_gesture.values()))
    ref_center = np.mean(ref_positions, axis=0)

    # Calculate the center of the detected landmarks
    detected_center = np.mean(landmark_positions, axis=0)

    # Calculate distances from the center for reference gesture
    ref_distances = np.linalg.norm(ref_positions - ref_center, axis=1)

    # Calculate distances from the center for detected landmarks
    detected_distances = np.linalg.norm(landmark_positions - detected_center, axis=1)

    # Check distance differences
    for i, (x, y) in predefined_gesture.items():
        distance_diff = np.linalg.norm(detected_distances[i] - ref_distances[i])
        if distance_diff > distance_threshold:
            return False

    # Optional: Check angles between key landmark pairs (e.g., index finger to thumb, etc.)
    if len(ref_positions) > 1:  # Make sure we have enough points to compare angles
        for i in range(len(ref_positions) - 1):
            ref_vector = ref_positions[i + 1] - ref_positions[i]
            detected_vector = landmark_positions[i + 1] - landmark_positions[i]

            # Calculate the angle between the vectors
            angle_diff = np.degrees(np.arccos(np.clip(np.dot(ref_vector, detected_vector) /
                                                      (np.linalg.norm(ref_vector) * np.linalg.norm(detected_vector)), -1.0, 1.0)))
            if angle_diff > angle_threshold:
                return False

    return True



# Start capturing the video stream.
cap = cv2.VideoCapture(0)
pTime = 0
cTime = 0
while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    flip = cv2.flip(image, 1)
    # Convert the BGR image to RGB.
    image_rgb = cv2.cvtColor(flip, cv2.COLOR_BGR2RGB)
    result = hands.process(image_rgb)
    flag = 0
    # Draw hand landmarks on the image.
    if result.multi_hand_landmarks:


        for hand_index, hand_landmarks in enumerate(result.multi_hand_landmarks):

            for id, lm in enumerate(hand_landmarks.landmark):

                h, w, c = flip.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                print(f"{id}:({cx},{cy}),")


            mp_drawing.draw_landmarks(flip, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            x_coords = [int(lm.x * flip.shape[1]) for lm in hand_landmarks.landmark]
            y_coords = [int(lm.y * flip.shape[0]) for lm in hand_landmarks.landmark]
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)

            # Draw rectangle around the hand
            cv2.rectangle(flip, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            # Determine the handedness of the hand
            hand_label = result.multi_handedness[hand_index].classification[0].label
            cv2.putText(flip, f"{hand_label} hand", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Check if the detected landmarks match the predefined gesture.
            if hand_label == "Left":
                if is_gesture_match(hand_landmarks.landmark, reference_gesture_1, image.shape):
                    control.PressKey(id_hex[2])
                    control.ReleaseKey(id_hex[4])
                else :
                    control.ReleaseKey(id_hex[2])
                if is_gesture_match(hand_landmarks.landmark, reference_gesture_2, image.shape):
                    control.PressKey(id_hex[4])
                    control.ReleaseKey(id_hex[2])
                else :
                    control.ReleaseKey(id_hex[4])
            elif hand_label == "Right":
                if is_gesture_match(hand_landmarks.landmark, reference_gesture_3, image.shape):
                    control.PressKey(id_hex[3])
                    control.ReleaseKey(id_hex[1])
                else :
                    control.ReleaseKey(id_hex[3])
                if is_gesture_match(hand_landmarks.landmark, reference_gesture_4, image.shape):
                    control.PressKey(id_hex[1])
                    control.ReleaseKey(id_hex[3])
                else :
                    control.ReleaseKey(id_hex[1])


    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(flip, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                (255, 0, 255), 3)
    # Display the image.
    cv2.imshow('Hand Tracking', flip)

    if cv2.waitKey(5) & 0xFF == 27:  # Press 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()
