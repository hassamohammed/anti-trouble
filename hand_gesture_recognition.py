import cv2
import mediapipe as mp
import time
import webbrowser

# Initialize the MediaPipe hand detection model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    # Get a frame from the webcam
    ret, frame = cap.read()

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect hands in the frame
    hand_detections = hands.process(rgb_frame)

    # Check if any hands were detected
    if hand_detections.multi_hand_landmarks:
        # Find the hand with the most fingers raised
        hand_index = 0  
        for i, hand_landmarks in enumerate(hand_detections.multi_hand_landmarks):
            if len(hand_landmarks.landmark) > len(hand_detections.multi_hand_landmarks[hand_index].landmark):
                hand_index = i

        # Check if the thumb is up
        thumb_up = True
        for i in range(4, 8):
            if hand_detections.multi_hand_landmarks[hand_index].landmark[i].x < hand_detections.multi_hand_landmarks[hand_index].landmark[i - 2].x:
                thumb_up = False
                break

        # If the thumb is up, open the URL of the video
        if thumb_up:
            print("thumb is up")
            webbrowser.open("https://youtube.com/@hassamohammed")
            time.sleep(5)

    # Display the frame
    cv2.imshow("Frame", frame)

    # Wait for a key press
    key = cv2.waitKey(1)

    # Quit if the Esc key is pressed
    if key == 27:
        break

# Release the webcam
cap.release()

# Close all windows
cv2.destroyAllWindows()
