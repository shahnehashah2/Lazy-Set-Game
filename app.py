# https://code.tutsplus.com/tutorials/creating-a-web-app-from-scratch-using-python-flask-and-mysql--cms-22972
from flask import Flask, render_template, request, redirect, url_for
import cv2
import numpy as np
import os
# from matplotlib import pyplot as plt
import sqlite3 as sql
import itertools

UPLOAD_FOLDER = 'tmp'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app = Flask(__name__, static_folder='tmp', static_url_path = '')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/")
def main(error=""):
    # Preprocess each image and convert then to specific sized rectangles
    # 266 x 200 (height x width) of resized image

    # ******************************* Uncomment when demoing***********
    # imagelist = os.listdir('train')
    #
    # for imageName in imagelist:
    #     # 3rd arg to imread specifies color or gray scale. >0 is color
    #     im = cv2.imread(os.path.join('train', imageName), 1)
    #     resizedImage = resizeImage(im)
    #     makeRectangle(resizedImage, 1, 'trained', imageName)

    return render_template('index.html', error=error)


def resizeImage(im):
    # im.shape is [height, width, RGB channels]
    ratio = 200.0/im.shape[1]
    dim = (200, int(im.shape[0] * ratio))
    resizedImage = cv2.resize(im, dim, interpolation = cv2.INTER_AREA)
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


@app.route("/solve", methods=['POST'])
def solve():
    # Get the uploaded file
    file = request.files['file']
    numOfCards = request.form['numOfCards']
    # Checking for no filename or unallowed extensions
    validateInput(file, numOfCards)
    numOfCards = int(numOfCards)

    dest = 'testcards'
    train_set = 'trained'
    # Empty destination folder
    empty(dest)
    # Read the uploaded image and convert all card images into rectangles
    im = cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'], file.filename), 1)
    makeRectangle1(im, numOfCards, dest)

    # Match the dest cards to ones in train_set and store the coded names of
    # the cards in coded_name array
    coded_name = find_matches(dest, train_set)
    # Find all possible combinations of 3 cards without repetition
    all_combi = list(itertools.combinations(coded_name, 3))
    sets = find_sets(all_combi)
    # Note- Images might be displayed rotated in browsers other than Firefox
    return render_template('solve.html', sets=sets,
                    file=url_for('static', filename=file.filename))


def validateInput(file, numOfCards):
    if file.filename == '':
        print('No selected file')
        # ***************** How do i send error msg with redirect?*****
        return redirect(url_for('main'))
    filename = file.filename
    if allowed_file(filename):
        if numOfCards.isdigit():
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        else:
            print('Number of cards must be numeric')
            return redirect(url_for('main'))
    else:
        print('Only JPEG or PNG files')
        return redirect(url_for('main'))

def allowed_file(f):
    return '.' in f and f.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def find_sets(all_combi):
    con = sql.connect('matches.db')
    cardDB = con.cursor()
    cardDB.execute('''CREATE TABLE IF NOT EXISTS Matches
                    (id INT, card1_code TEXT, card2_code TEXT,
                    card3_code TEXT, card1 TEXT, card2 TEXT, card3 TEXT)''')
    # Delete previous entries of the table, if any, and vacuum from memory
    cardDB.execute("DELETE FROM Matches")
    cardDB.execute("VACUUM")

    count = 0
    for i in all_combi:
        if (int(i[0][0]) + int(i[1][0]) + int(i[2][0])) % 3 == 0:
            if (int(i[0][1]) + int(i[1][1]) + int(i[2][1])) % 3 == 0:
                if (int(i[0][2]) + int(i[1][2]) + int(i[2][2])) % 3 == 0:
                    if (int(i[0][3]) + int(i[1][3]) + int(i[2][3])) % 3 == 0:
                        count += 1
                        items = [reverse_dict(j) for j in i]
                        cardDB.execute("INSERT INTO Matches VALUES (?,?,?,?,?,?,?)",\
                                        (count, i[0], i[1], i[2], items[0], items[1], items[2]))
    con.commit()
    cur = con.execute("SELECT card1, card2, card3 from Matches")
    sets = []
    for row in cur:
       sets.append([row[0], row[1], row[2]])
    if len(sets)>0:
        return sets
    else:
        return "No set found"

def reverse_dict(item):
    con = sql.connect('testcards.db')
    with con:
        cardDB = con.cursor()
        cardDB.execute("SELECT name from TestCards WHERE code_name = :item",\
                            {"item": item})
        row = cardDB.fetchone()
    return row[0]

def find_matches(dest, train_set):
    testlist = os.listdir(dest)
    imagelist = os.listdir(train_set)
    id = 0
    coded_name = []
    con = sql.connect('testcards.db')
    # cur = con.execute("DROP TABLE TestCards")
    cardDB = con.cursor()
    cardDB.execute('''CREATE TABLE IF NOT EXISTS TestCards
                    (id INT, name TEXT, shape TEXT, repeat INT,
                    fill TEXT, color TEXT, code_name TEXT)''')
    # Delete previous entries of the table, if any, and vacuum from memory
    cardDB.execute("DELETE FROM TestCards")
    cardDB.execute("VACUUM")

    dict_name = {'P':'1', 'G':'2', 'R':'3', 'D':'1', 'O':'2', 'W':'3',
                        'E':'1', 'F':'2', 'S':'3'}

    for im1 in testlist:
        image1 = cv2.imread(os.path.join(dest, im1), 1)
        img2 = ''
        bestMatch = 4000000
        for im2 in imagelist:
            image2 = cv2.imread(os.path.join('trained', im2), 1)
            # Calculate per elements difference between two arrays
            diff = cv2.absdiff(preprocess(image1),preprocess(image2))
            # Setting a high sigma leads to false matches.
            # Setting too low leads to false mismatches
            # diff = cv2.GaussianBlur(diff,(5,5), 2)
            # flag, thresh = cv2.threshold(diff, 200, 255, cv2.THRESH_BINARY)
            # cv2.imshow('thresh', thresh)
            # cv2.waitKey(0)
            # print (im1, im2, np.sum(diff))
            # Find the images with minimum difference to get the best match
            if(np.sum(diff) < bestMatch):
                bestMatch = np.sum(diff)
                img2 = im2
        # Ignore the card with no match
        if img2 == '':
            print("No match")
            continue
        # Add the matched card into database
        id += 1

        # It is difficult to distinguish between full and empty fill.
        # So explicit check is made in scenario whether fill is not stripe
        fill = img2[2]
        if fill != 'S':
            fill = get_fill(image1)
        color = get_color(image1)
        # Get the shape and number of repeats in shape from name of training card
        shape = img2[0]
        repeat = img2[1]

        #Storing the name for easy retrieval
        name = shape+str(repeat)+fill+color
        name1 = dict_name[shape]+str(repeat)+dict_name[fill]+dict_name[color]
        coded_name.append(name1)
        cardDB.execute("INSERT INTO TestCards VALUES (?,?,?,?,?,?,?)",\
                        (id, name, shape, fill, repeat, color, name1))
    # cur = con.execute("SELECT id, name, shape, fill, repeat, color, code_name from TestCards")
    # for row in cur:
    #    print (row)
    con.commit()
    return coded_name


def empty(dest):
    testlist = os.listdir(dest)
    for f in testlist:
        os.unlink(os.path.join(dest, f))

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

    # img1 = im.copy()
    # cv2.drawContours(img1, contours, -1, (255,0,0), 3)
    # cv2.namedWindow('contours', flags= cv2.WINDOW_NORMAL)
    # cv2.imshow('contours', img1)
    # cv2.waitKey(0)

    i = 1
    for c in contours:
        # Calculate the perimeter
        peri = cv2.arcLength(c, True)
        # For contour c, approximate the curve based on calculated perimeter.
        # 2nd arg is accuracy, 3rd arg states that the contour curve is closed
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        # Create an array of floats of desired image dimension
        h = np.array([ [0, 200],[0,0],[266, 0],[266, 200] ], np.float32)
        # Gotta change the approx data set also to float32
        approx = approx.astype(np.float32, copy=False)

        # Check whether approx is in portrait mode. If not, change from landscape to portrait
        x1 = approx[0][0][0]
        y1 = approx[0][0][1]
        x2 = approx[1][0][0]
        y2 = approx[1][0][1]
        x3 = approx[2][0][0]
        y3 = approx[2][0][1]

        # Get the distance squared of top edge and left edge
        l1 = ((x1-x2) ** 2) + ((y1-y2) ** 2)
        l2 = ((x2-x3) ** 2) + ((y2-y3) ** 2)

        if l2<l1:
            # Shift the array once clockwise
            approx = shift(approx)

        #Transform the approx data array to h
        transform = cv2.getPerspectiveTransform(approx,h)

        # for j,x in enumerate(approx[0]):
            # print(j,x.tolist())
            # cv2.putText(im, str(j), tuple(int(y) for y in x.tolist()), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))
        # cv2.namedWindow('contours', flags= cv2.WINDOW_NORMAL)
        # cv2.imshow('contours', im)
        # cv2.waitKey(0)
        # Apply the transformed perspective to original image
        warp = cv2.transpose(cv2.warpPerspective(im,transform,(266,200)))
        # warp = cv2.warpPerspective(im,transform,(266,200))
        # Rotate image by 90 degrees
        cv2.imwrite(os.path.join(folder, str(i)+'.jpg'), warp)
        i += 1


def shift(seq):
    temp = seq[3].copy()
    for i in range(3, 0, -1):
        seq[i] = seq[i-1]
    seq[0] = temp
    return seq


def preprocess(im):
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),2 )
    thresh = cv2.adaptiveThreshold(blur,255,1,1,11,1)
    blur_thresh = cv2.GaussianBlur(thresh,(5,5),5)
    return blur_thresh


def get_color(im):
    height = 266
    width = 200

    # Color is returned in BGR format
    for i in range(height):
        for j in range(width):
            bgr = im[i][j]
            if sum(bgr) > 500:
                im[i][j] = [0,0,0]

    # To count the non-black pixels in image, create a grayscale copy of it
    # np.mean averages over all pixels (including black)
    im1 = cv2.cvtColor( im, cv2.COLOR_RGB2GRAY )
    bgr_mean = (np.mean(im, axis=(0,1)) * im1.size) // np.count_nonzero(im1)

    color=''
    blue = bgr_mean[0]
    green = bgr_mean[1]
    red = bgr_mean[2]

    if red>blue and red> green and red-green>50:
        color = 'R'
    elif green>blue and green>red and green-red>50:
        color = 'G'
    else:
        color = 'P'
    return color



def get_fill(im):
    # im should be only a full or empty card, not striped
    # Set Region of Interest to a single line passing vertically through
    # top half of card
    col = 100
    rows = range(0,133)

    countColor = 0
    countWhite = 0

    for i in rows:
        bgr = im[i][col]
        blue = bgr[0]
        green = bgr[1]
        red = bgr[2]
        # White or close to white pixel has RGB values over 150 each
        if (int(blue) + int(green) + int(red)) > 450:
            countWhite += 1
        else:
            countColor += 1

    if (countWhite//countColor) > 10:
        return 'E'
    else:
        return 'F'


if __name__ == "__main__":
    app.run()
