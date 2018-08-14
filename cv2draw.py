import numpy as np
import cv2
img = cv2.imread("image-1.jpg")
cv2.rectangle(img,(257,148),(482,408),(0,255,0),3)
cv2.imshow("img", img)
cv2.waitKey(0)