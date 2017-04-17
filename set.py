# Preprocessing - Standardize the height and make each train image rectangle
# within the color folders
# Get game image from user (game1)
# game2 = copy of game image
#
# On game1, do the following:
#     Turn it to grayscale
#     canny edge detector
#     contours
#
#     For each contoured card (card1):
#        region of interest
#        Use this region of interest and map it onto game2. Get color using this
#        Open test image folder for this color. For each image inside (test1):
#            Subtract and find difference between card1 and test1.
#            The ones= with lesser than threshold is the matching card

import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import sqlite3 as sql


def resizeImage(im):
    # Creating a named window to fit image to normal full window size
    # # cv2.namedWindow('original', flags=cv2.WINDOW_NORMAL)
    # cv2.imshow('original', im)
    # cv2.waitKey(0)
    # print (im.shape)
    # im.shape is [height, width, RGB components]
    ratio = 200.0/im.shape[1]
    dim = (200, int(im.shape[0] * ratio))
    resizedImage = cv2.resize(im, dim, interpolation = cv2.INTER_AREA)
    print ('Image size=', resizedImage.shape) # Prints tuple (rows, cols, channels)
    return resizedImage

def makeRectangle(im, numOfCards, folder, imageName):
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    # Removing noise from the image using blur
    blur = cv2.GaussianBlur(gray,(1,1),100)
    # flag, thresh = cv2.threshold(blur, 120, 255, cv2.THRESH_BINARY)
    # Using Canny Edge detection
    edged = cv2.Canny(blur, 30, 200)

    # Highlight all the contours in the image
    _, contours, _ = cv2.findContours(edged,
                                cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    # Sort the contours by area so we can get the outside rectangle contour
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:numOfCards]
    for c in contours:
        # Calculate the perimeter
        # c = contours[numOfCards-1]
        peri = cv2.arcLength(c, True)
        # For contour c, approximate the curve based on calculated perimeter.
        # 2nd arg is accuracy, 3rd arg states that the contour curve is closed
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        # Get the rectangle enclosing the points specified in arg
        # rect = cv2.minAreaRect(c)
        # Find the vertices of the rectangle
        # r = cv2.boxPoints(rect)

        # Create an array of floats of desired image dimension
        h = np.array([ [0,0],[266,0],[266,200],[0,200] ], np.float32)
        # Gotta change the approx data set also to float32
        approx = approx.astype(np.float32, copy=False)

        # print(approx.dtype, h.dtype)
        #Transform the approx data array to h
        transform = cv2.getPerspectiveTransform(approx,h)

        # Apply the transformed perspective to original image
        warp = cv2.transpose(cv2.warpPerspective(im,transform,(266,200)))
        # Rotate image by 90 degrees
        cv2.imwrite(os.path.join(folder, imageName), warp)


# Preprocess each image and convert then to specific sized rectangles
# 266 x 200 (height x width) of resized image
# imagelist = os.listdir('green')
#
# for imageName in imagelist:
#     # 3rd arg to imread specifies color or gray scale. >0 is color
#     im = cv2.imread(os.path.join('green', imageName), 1)
#     resizedImage = resizeImage(im)
#     makeRectangle(resizedImage, 1, 'green1', imageName)


def makeRectangle1(im, numOfCards, folder):
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    # Removing noise from the image using blur
    blur = cv2.GaussianBlur(gray,(1,1),100)
    flag, thresh = cv2.threshold(blur, 120, 255, cv2.THRESH_BINARY)
    # Using Canny Edge detection - not detecting all edges properly
    # edged = cv2.Canny(blur, 30, 300)
    #
    # plt.subplot(121), plt.imshow(im, cmap='gray')
    # plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    # plt.subplot(122), plt.imshow(edged, cmap='gray')
    # plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    # plt.show()

    # Highlight all the contours in the image
    _, contours, _ = cv2.findContours(thresh,
                                cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)


    # Sort the contours by area so we can get the outside rectangle contour
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:numOfCards]

    img1 = im.copy()
    cv2.drawContours(img1, contours, -1, (255,0,0), 3)
    cv2.namedWindow('contours', flags= cv2.WINDOW_NORMAL)
    cv2.imshow('contours', img1)
    cv2.waitKey(0)

    i = 0
    for c in contours:
        # Calculate the perimeter
        peri = cv2.arcLength(c, True)
        # For contour c, approximate the curve based on calculated perimeter.
        # 2nd arg is accuracy, 3rd arg states that the contour curve is closed
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        # rect = cv2.minAreaRect(c)
        # # Find the vertices of the rectangle
        # r = cv2.boxPoints(rect)
        # print(r)

        # Create an array of floats of desired image dimension
        h = np.array([ [0, 200],[0,0],[266, 0],[266, 200] ], np.float32)
        # Gotta change the approx data set also to float32
        approx = approx.astype(np.float32, copy=False)

        #Transform the approx data array to h
        transform = cv2.getPerspectiveTransform(approx,h)

        # Apply the transformed perspective to original image
        warp = cv2.transpose(cv2.warpPerspective(im,transform,(266,200)))
        # Rotate image by 90 degrees
        cv2.imwrite(os.path.join(folder, str(i)+'.jpg'), warp)
        i += 1


# Empty destination folder
dest = 'testcards'
# for f in os.listdir(dest):
#     os.unlink(os.path.join(dest, f))
#
# im = cv2.imread(os.path.join('testimage4.jpg'), 1)
# makeRectangle1(im, 12, dest)


def findDifference(im):
    # Are images well-aligned?
    # If not, you may want to run cross-correlation first, to find the best alignment first.
    # SciPy has functions to do it.
    #
    # Is exposure of the images always the same? (Is lightness/contrast the same?)
    # If not, you may want to normalize images.
    # But be careful, in some situations this may do more wrong than good.
    # For example, a single bright pixel on a dark background will make the normalized image very different.
    #
    # Is color information important?
    # If you want to notice color changes, you will have a vector of color values per point,
    # rather than a scalar value as in gray-scale image. You need more attention when writing such code.
    pass

def preprocess(im):
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),2 )
    thresh = cv2.adaptiveThreshold(blur,255,1,1,11,1)
    blur_thresh = cv2.GaussianBlur(thresh,(5,5),5)
    return blur_thresh

imagelist = os.listdir('green1')
testlist = os.listdir(dest)
id = 0

con = sql.connect('testcards.db')

for im1 in testlist:
    image1 = cv2.imread(os.path.join(dest, im1), 1)
    if image1 is not None:
        for im2 in imagelist:
            image2 = cv2.imread(os.path.join('green1', im2), 1)
            if image2 is not None:
                # Calculate per elements difference between two arrays
                diff = cv2.absdiff(preprocess(image1),preprocess(image2))
                # Setting a high sigma leads to false matches.
                # Setting too low leads to false mismatches
                diff = cv2.GaussianBlur(diff,(5,5), 2)
                flag, thresh = cv2.threshold(diff, 200, 255, cv2.THRESH_BINARY)
                cv2.imshow('thresh', thresh)
                # cv2.waitKey(0)
                # print (im1, im2, np.sum(thresh))
                # Set a threshold for match
                if(np.sum(thresh) < 3500):
                    id += 1
                    with con:
                        cardDB = con.cursor()
                        cardDB.execute('''CREATE TABLE IF NOT EXISTS TestCards
                                        (id INT, name TEXT, shape TEXT,
                                        fill TEXT, repeat INT, color TEXT)''')
                        cardDB.execute("INSERT INTO TestCards VALUES (?,?,?,?,?,?)",\
                                        (id, im2[0:4], im2[0], im2[2], im2[1], im2[3]))

cur = con.execute("SELECT id, name, shape, fill, repeat, color from TestCards")
for row in cur:
   print (row)
cur = con.execute("DROP TABLE TestCards")
