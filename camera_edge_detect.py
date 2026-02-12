import cv2

# Read the images captured by the camera (default /dev/video11 on rk3588)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)

        cv2.imshow('Edge Detection', edges)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
