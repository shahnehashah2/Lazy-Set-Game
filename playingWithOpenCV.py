# http://docs.opencv.org/3.2.0/d6/d00/tutorial_py_root.html

import cv2
import sys
import numpy as np

print(cv2.__version__) # This is 3.2.0

img = cv2.imread("setTrain.jpg")  ## Read image file in color

if (img == None):                      ## Check for invalid input
    print ("Could not open or find the image")
else:
    cv2.namedWindow('Display Window')        ## create window for display
    cv2.imshow('Display Window',img)         ## Show image in the window
    print ("size of image: ") ,img.shape        ## print size of image
    cv2.waitKey(0)                           ## Wait for keystroke
    #cv2.destroyAllWindows()                  ## Destroy all windows

alpha = 0.5
beta = 0.5
gamma = 0
img2 = cv2.imread("dog.jpg");

superImposedImage = cv2.addWeighted(img, alpha, img, beta, gamma)
cv2.imshow('superImposed', superImposedImage)
cv2.waitKey(0)


newimg = img - img ## Should be of the same size to work. If images are same, newimg will have all black pixels
cv2.imshow('NewIMg', newimg)
cv2.waitKey(0)


# Harris Corner Detection - works only on grayscale and floast32 images
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)
dst = cv2.cornerHarris(gray,2,3,0.04)
#result is dilated for marking the corners, not important
dst = cv2.dilate(dst,None)
# Threshold for an optimal value, it may vary depending on the image.
img[dst>0.01*dst.max()]=[0,0,255]
cv2.imshow('dst',img)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()


# Oriented FAST and Rotated BRIEF corner detection
from matplotlib import pyplot as plt

# Initiate ORB detector
orb = cv2.ORB_create()
# find the keypoints with ORB
kp = orb.detect(img,None)
# compute the descriptors with ORB
kp, des = orb.compute(img, kp)
# draw only keypoints location,not size and orientation
imgORB = cv2.drawKeypoints(img, kp, None, color=(0,255,0), flags=0)
plt.imshow(imgORB), plt.show()
