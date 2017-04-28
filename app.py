from flask import Flask, render_template, request, redirect, url_for, flash
import cv2
import numpy as np
import os
import sqlite3 as sql
import itertools

UPLOAD_FOLDER = '.'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

# Static folder needs to be specified because flask needs it to look for images in jinja2
# URL path is empty string so we don't have to provide absolute path
app = Flask(__name__, static_folder='.', static_url_path = '')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# Set session secret key
app.secret_key = 'some_secret'


@app.route("/rules", methods=['GET'])
def rules():
    return render_template('rules.html')


@app.route("/")
def main():
    # Preprocess each image and convert then to specific sized rectangles
    # 266 x 200 (height x width) of resized image
    print('$$$$$$$$$$$$$$$$$$$reaches here***************')
    imagelist = os.listdir('train')
    # Empty the trained folder to re-train

    #***************Uncomment before demoing******
    # Needs to be run the first time this program is executed on a computer
    # empty('trained')
    # for imageName in imagelist:
    #     # 3rd arg to imread specifies color or gray scale. >0 is color
    #     im = cv2.imread(os.path.join('train', imageName), 1)
    #     resizedImage = resizeImage(im)
    #     makeRectangle(resizedImage, 1, 'trained', 'doTrain', imageName)
    return render_template('index.html')


@app.route("/solve", methods=['POST'])
def solve():
    # Get the uploaded file
    file = request.files['file']
    numOfCards = request.form['numOfCards']
    # Checking for no filename or unallowed extensions
    error = validateInput(file, numOfCards)
    if error:
        flash(error)
        # Render and do not redirect to avoid re-processing of training images
        return render_template('index.html')
    numOfCards = int(numOfCards)

    dest = 'testcards'
    train_set = 'trained'
    # Empty destination folder
    empty(dest)
    # Read the uploaded image and convert all card images into rectangles
    im = cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'], file.filename), 1)
    makeRectangle(im, numOfCards, dest, 'doTest')

    # Match the dest cards to ones in train_set and store the coded names of
    # the cards in coded_name array
    coded_name = find_matches(dest, train_set)
    # Find all possible combinations of 3 cards without repetition
    all_combi = list(itertools.combinations(coded_name, 3))
    sets = find_sets(all_combi)
    if sets == "None":
        found = 0
    else:
        found = len(sets)
    sets_images = get_imageNames(sets)
    # Note- Images might be displayed rotated in browsers other than Firefox
    return render_template('solve.html', found=found, sets=sets, sets_images=sets_images,
                    file=url_for('static', filename=file.filename))


def resizeImage(im):
    # im.shape is [height, width, RGB channels]
    ratio = 200.0/im.shape[1]
    dim = (200, int(im.shape[0] * ratio))
    resizedImage = cv2.resize(im, dim, interpolation = cv2.INTER_AREA)
    return resizedImage


def makeRectangle(im, numOfCards, folder, trainOrTest, imageName=''):
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    # Removing noise from the image using blur
    blur = cv2.GaussianBlur(gray,(1,1),100)
    flag, thresh = cv2.threshold(blur, 120, 255, cv2.THRESH_BINARY)
    # Using Canny Edge detection
    if trainOrTest == 'doTrain':
        edged = cv2.Canny(blur, 30, 200)
    else:
        edged = thresh

    # Highlight all the contours in the image
    _, contours, _ = cv2.findContours(edged,
                                cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    # Sort the contours by area so we can get the outside rectangle contour
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:numOfCards]

    # Counter to keep a tab of number of images for naming purposes
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

        if l2 < l1:
            # Shift the array once clockwise
            approx = shift(approx)

        #Transform the approx data array to h
        transform = cv2.getPerspectiveTransform(approx,h)

        # Apply the transformed perspective to original image
        warp = cv2.transpose(cv2.warpPerspective(im,transform,(266,200)))

        # The naming of files varies based on whether it is training image or test image
        if trainOrTest == 'doTrain':
            imName = imageName
        else:
            imName = str(i)+'.jpg'
            i += 1
        cv2.imwrite(os.path.join(folder, imName), warp)


# Check whether a valid file and a valid number of cards are entered
def validateInput(file, numOfCards):
    error = ''
    if file.filename == '':
        error = 'No selected file'
    filename = file.filename
    if allowed_file(filename):
        if numOfCards.isdigit() and int(numOfCards) > 2:
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        else:
            # There must be atleast 3 cards to check for a set
            error = 'Number of cards must be numeric > 3'
    else:
        error = 'Only JPEG or PNG files'
    return error


# Check whether the uploaded file is a jpg or png image
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
        # ********** Make sure this is working************
        return "None"

# Get the actual names of images in a set
# (like [D3EG.jpg, D2EG.jpg, D1EG.jpg]) for retrieving the images from the
# test folder. Map the image names from their codenames in database
def get_imageNames(sets):
    con = sql.connect('testcards.db')
    cardDB = con.cursor()

    setsName = []
    for set in sets:
        setName = []
        for i in set:
            cardDB.execute("SELECT idName from TestCards WHERE name = :i",\
                        {"i": i})
            row = cardDB.fetchone()
            setName.append(row[0])
        setsName.append(setName)
    return setsName


# Get the alphabetic code name from numeric code name
def reverse_dict(item):
    con = sql.connect('testcards.db')
    with con:
        cardDB = con.cursor()
        cardDB.execute("SELECT name from TestCards WHERE code_name = :item",\
                            {"item": item})
        row = cardDB.fetchone()
    return row[0]

# Most of the magic happens here. Match each uploaded card with 'trained' images
def find_matches(dest, train_set):
    testlist = os.listdir(dest)
    imagelist = os.listdir(train_set)
    id = 0
    coded_name = []
    con = sql.connect('testcards.db')
    cardDB = con.cursor()
    cardDB.execute("DROP TABLE IF EXISTS TestCards")
    cardDB.execute('''CREATE TABLE TestCards
                    (id INT, name TEXT, idName TEXT, code_name TEXT)''')

    # Need this dictionary to lookup codename for use in logic to find sets
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
        # So explicit check is made in scenario where fill type is not stripe(S)
        fill = img2[2]
        if fill != 'S':
            fill = get_fill(image1)
        # Find the color of the card
        color = get_color(image1)
        # Get the shape and number of repeats in shape from name of training card
        shape = img2[0]
        repeat = img2[1]

        # Storing in database with two names (D2EG and 1212)
        name = shape + str(repeat) + fill + color
        name1 = dict_name[shape]+str(repeat)+dict_name[fill]+dict_name[color]
        # Storing the numeric coded name in an array for applying logic to
        # finding sets. The name in database will be later used to lookup
        # image of the card which is a part of found sets
        coded_name.append(name1)
        cardDB.execute("INSERT INTO TestCards VALUES (?,?,?,?)",\
                        (id, name, im1, name1))
    cur = con.execute("SELECT id, name, idName, code_name from TestCards")
    for row in cur:
       print (row)
    con.commit()
    return coded_name


# Empty the specified folder. Called everytime the game is restarted
def empty(dest):
    listname = os.listdir(dest)
    for f in listname:
        os.unlink(os.path.join(dest, f))


# Shift the array elements one time
def shift(seq):
    temp = seq[3].copy()
    for i in range(3, 0, -1):
        seq[i] = seq[i-1]
    seq[0] = temp
    return seq


# Image needs to be preprocessed before comparison via absdiff()
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

    # np.mean averages over all pixels (including black)
    # To count the non-black pixels in image, create a grayscale copy of it
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
