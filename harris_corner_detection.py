#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

import cv2
import numpy as np

# Read image
img = cv2.imread('./pic1.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Harris corner detection
gray = np.float32(gray)
dst = cv2.cornerHarris(gray, 2, 3, 0.08)
dst = cv2.dilate(dst, None)
img[dst > 0.01 * dst.max()] = [0, 0, 255]

# Display image
cv2.imshow('Harris Corner Detection', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
