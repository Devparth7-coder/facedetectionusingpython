import cv2
import numpy as np
from deepface import DeepFace

def detect_faces():
    # Ask for the user's name
    user_name = input("Please enter your name: ")

    # Load the pre-trained Haar cascade classifier for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Start video capture (0 for the default camera)
    video_capture = cv2.VideoCapture(0)

    if not video_capture.isOpened():
        print("Error: Could not access the camera.")
        return

    print("Press 'c' to capture photo.")

    while True:
        # Capture a single frame
        ret, frame = video_capture.read()

        if not ret:
            print("Error: Failed to capture image.")
            break

        # Convert the frame to grayscale (required for Haar cascade)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Draw rectangles around detected faces and show name, age, and emotion
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Crop the detected face
            face_crop = frame[y:y+h, x:x+w]

            # Analyze the face for emotion and age
            try:
                analysis = DeepFace.analyze(face_crop, actions=['age', 'emotion'], enforce_detection=False)
                if isinstance(analysis, list):
                    analysis = analysis[0]  # Access the first result if it's a list
                age = analysis.get('age', 'N/A')
                emotion = analysis.get('dominant_emotion', 'N/A')

                # Display name, age, and emotion on the frame
                cv2.putText(frame, f"Name: {user_name}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(frame, f"Age: {age}", (x, y - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(frame, f"Emotion: {emotion}", (x, y - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            except Exception as e:
                print(f"Analysis failed: {e}")

        # Display the frame with detected faces
        cv2.imshow('Face Detection', frame)

        # Check for user input
        key = cv2.waitKey(1) & 0xFF

        if key == ord('c') and len(faces) > 0:
            # Save the captured face
            file_path = f"captured_{user_name}.jpg"
            cv2.imwrite(file_path, face_crop)
            print(f"Photo saved as {file_path}")
            break

    # Release the video capture and close windows
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_faces()
