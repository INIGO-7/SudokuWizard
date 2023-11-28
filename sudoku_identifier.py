import cv2 as cv
import os
from imutils import contours
import numpy as np
from imutils.perspective import four_point_transform
from skimage.segmentation import clear_border
import imutils

def find_puzzle(image, debug=False):
    # convert the image to grayscale, and apply an adaptative threshold
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    thresh = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 57, 12)

    if debug:
        cv.imshow("first thresh", thresh)
        cv.waitKey(0)

    # Filter out all numbers and noise to isolate only boxes
    cnts = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        area = cv.contourArea(c)
        if area < 1200:
            cv.drawContours(thresh, [c], -1, (0,0,0), -1)

    if debug:
        cv.imshow("Puzzle Thresh", thresh)
        cv.waitKey(0)
    
    # Make white lines thicker in order to better identify squares
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))  # adjust the kernel size
    dilated = cv.dilate(thresh, kernel, iterations=1)  # adjust the number of iterations

    if debug:
        cv.imshow("Puzzle dilated", dilated)
        cv.waitKey(0)

    # find contours in the thresholded image and sort them by size in
    # descending order
    cnts = cv.findContours(dilated.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv.contourArea, reverse=True)
    # initialize a contour that corresponds to the puzzle outline
    puzzleCnt = None
    # loop over the contours
    for c in cnts:
        # approximate the contour
        peri = cv.arcLength(c, True)
        approx = cv.approxPolyDP(c, 0.02 * peri, True)
        # if our approximated contour has four points, then we can
        # assume we have found the outline of the puzzle
        if len(approx) == 4:
            puzzleCnt = approx
            break

    # if the puzzle contour is empty then our script could not find
    # the outline of the Sudoku puzzle so raise an error
    if puzzleCnt is None:
        raise Exception(("Could not find Sudoku puzzle outline. "
        "Try debugging your thresholding and contour steps."))
    
    # check to see if we are visualizing the outline of the detected
    # Sudoku puzzle
    output = image.copy()
    cv.drawContours(output, [puzzleCnt], -1, (0, 255, 0), 2)
    if debug:
        cv.imshow("Puzzle Outline", output)
        cv.waitKey(0)

    # apply a four point perspective transform to both the original
    # image and grayscale image to obtain a top-down bird's eye view
    # of the puzzle
    puzzle = four_point_transform(image, puzzleCnt.reshape(4, 2))
    dilated = four_point_transform(dilated, puzzleCnt.reshape(4, 2))

    if debug:
        cv.imshow("Puzzle Transform", puzzle)
        cv.waitKey(0)

    # return a 2-tuple of puzzle in both RGB and grayscale
    return (puzzle, dilated)

def get_sudoku_squares(thresh, puzzle, debug=False):

    # Fix horizontal and vertical lines
    vertical_kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, 3))
    thresh = cv.morphologyEx(thresh, cv.MORPH_CLOSE, vertical_kernel, iterations=10)

    horizontal_kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 1))
    thresh = cv.morphologyEx(thresh, cv.MORPH_CLOSE, horizontal_kernel, iterations=10)

    if debug:
        cv.imshow("horizontal an vertical lines fixed", thresh)
        cv.waitKey(0)

    # Sort by top to bottom and each row by left to right
    invert = 255 - thresh
    cnts = cv.findContours(invert, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    (cnts, _) = contours.sort_contours(cnts, method="top-to-bottom")

    sudoku_rows = []
    row = []
    for (i, c) in enumerate(cnts, 1):
        area = cv.contourArea(c)
        if area < 50000:
            row.append(c)
            if i % 9 == 0:  
                (cnts, _) = contours.sort_contours(row, method="left-to-right")
                sudoku_rows.append(cnts)
                row = []

    # Iterate through each box
    sudoku_boxes = []

    for row in sudoku_rows:
        for c in row:
            mask = np.zeros(puzzle.shape, dtype=np.uint8)
            cv.drawContours(mask, [c], -1, (255,255,255), -1)
            result = cv.bitwise_and(puzzle, mask)
            result[mask==0] = 255
            sudoku_boxes.append(result)
    
    return (invert, thresh, sudoku_boxes)

# Load image
image = cv.imread('res/photos/sudoku/sudokuLibro1.jpeg')
templates = [cv.imread(f'res/photos/numbers/number{i}.jpeg') for i in range(1, 10)]

cv.imshow("original_image", image)
cv.waitKey(0)

puzzle, thresh = find_puzzle(image, debug=False)

invert, new_thresh, results = get_sudoku_squares(thresh, puzzle, debug=False)

def get_number(img, templates):

    # con umbralización del color
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    _, thresh = cv.threshold(gray, 100, 255, cv.THRESH_BINARY)

    if np.all(thresh == 255):
        return 0
    else:
        results = {'1':0, '2':0, '3':0, '4':0, '5':0, '6':0, '7':0, '8':0, '9':0}

        for idx, template in enumerate(templates):
            result = cv.matchTemplate(img, template, cv.TM_CCOEFF_NORMED)
            # Find the position of the best match
            _, max_val, _, _ = cv.minMaxLoc(result)
            results[idx + 1] = max_val
        
        return max(results, key=results.get)

sudoku_arr = []
for cell in results:
    sudoku_arr.append(get_number(cell, templates))

print(np.array(sudoku_arr).reshape(9, 9))

# HACER RESHAPE DEL ARRAY

# METER A DATAFRAME DE PANDAS