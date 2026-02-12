
import cv2
import numpy as np

# Read the image
img = cv2.imread('./pic1.jpg')

# Convert to greyscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Using Shi-Tomasi corner detection
corners = cv2.goodFeaturesToTrack(gray, maxCorners=100, qualityLevel=0.01, minDistance=10, blockSize=3)

# If a corner point is detected, it is converted to integer format and plotted
if corners is not None:
    corners = np.int8(corners)
    for corner in corners:
        x, y = corner.ravel()
        cv2.circle(img, (x, y), 3, 255, -1)

# Display image
cv2.imshow('Shi-Tomasi Corner Detection', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
