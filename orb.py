
import cv2
import numpy as np

# Read the image
img = cv2.imread('pic1.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Create ORB objects
orb = cv2.ORB_create()

# Detect keypoints and compute descriptors
kp, des = orb.detectAndCompute(gray, None)

# Mapping key points
img = cv2.drawKeypoints(gray, kp, img, color=(0,255,0), flags=0)

# Display image
cv2.imshow('ORB', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
