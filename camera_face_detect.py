import cv2

# Load pre-trained Haar feature classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Read the images captured by the camera(default /dev/video11 on rk3588)
cap = cv2.VideoCapture(0)

# Loop through each frame
while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # If the read is successful, the image is converted to greyscale
    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in images using cascade classifiers
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        # Draw a rectangle around each detected face
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            print("detect a face")

        # Display image
        cv2.imshow('Face Detection', frame)

        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
