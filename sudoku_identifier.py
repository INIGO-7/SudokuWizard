import cv2 as cv
import pandas as pd
import numpy as np
import os
from exceptions import SudokuDetectionError

import imutils
from imutils.perspective import four_point_transform
from imutils import contours
from skimage.segmentation import clear_border


NUMBER_TEMPLATES = [cv.imread(f'res/photos/numbers/number{i}HQ_nomargin.jpg') for i in range(1, 10)]

def find_puzzle(image, verbose=False):
    # convert the image to grayscale, and apply an adaptative threshold
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    thresh = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 57, 12)

    if verbose:
        cv.imshow("first thresh", thresh)
        cv.waitKey(0)

    # Make white lines thicker in order to better identify squares
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (2, 2))  # adjust the kernel size
    dilated = cv.dilate(thresh, kernel, iterations=1)  # adjust the number of iterations

    if verbose:
        cv.imshow("First dilatacioneishon", dilated)
        cv.waitKey(0)

    # Morphological closing to connect lines
    closing_kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    thresh = cv.morphologyEx(dilated, cv.MORPH_CLOSE, closing_kernel)

    if verbose:
        cv.imshow("closed", thresh)
        cv.waitKey(0)

    # Blur to reduce noise
    thresh = cv.medianBlur(thresh, 3)

    if verbose:
        cv.imshow("blur antes de buscar contornos", thresh)
        cv.waitKey(0)

    # Filter out all numbers and noise to isolate only boxes
    cnts = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        area = cv.contourArea(c)
        if area < 1200:
            cv.drawContours(thresh, [c], -1, (0,0,0), -1)

    if verbose:
        cv.imshow("Contornos encontrados", thresh)
        cv.waitKey(0)

    # Morphological closing to connect lines
    closing_kernel = cv.getStructuringElement(cv.MORPH_RECT, (9, 9))
    thresh = cv.morphologyEx(thresh, cv.MORPH_CLOSE, closing_kernel)

    if verbose:
        cv.imshow("big closing bro", thresh)
        cv.waitKey(0)

    # Probamos a arreglar las lineas horizontales y verticales!
    vertical_kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, 3))
    thresh = cv.morphologyEx(thresh, cv.MORPH_CLOSE, vertical_kernel, iterations=10)

    horizontal_kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 1))
    thresh = cv.morphologyEx(thresh, cv.MORPH_CLOSE, horizontal_kernel, iterations=10)

    if verbose:
        cv.imshow("Horiz and vert FIXED", thresh)
        cv.waitKey(0)

    # find contours in the thresholded image and sort them by size in
    # descending order
    cnts = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
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
        "Try verboseging your thresholding and contour steps."))
    
    # check to see if we are visualizing the outline of the detected
    # Sudoku puzzle
    output = image.copy()
    cv.drawContours(output, [puzzleCnt], -1, (0, 255, 0), 2)
    if verbose:
        cv.imshow("Puzzle Outline", output)
        cv.waitKey(0)

    # apply a four point perspective transform to both the original
    # image and grayscale image to obtain a top-down bird's eye view
    # of the puzzle
    puzzle = four_point_transform(image, puzzleCnt.reshape(4, 2))
    thresh = four_point_transform(thresh, puzzleCnt.reshape(4, 2))

    if verbose:
        cv.imshow("Puzzle Transform", puzzle)
        cv.waitKey(0)
    
    # return a 2-tuple of puzzle in both RGB and grayscale
    
    return (puzzle, thresh)

def get_sudoku_squares(thresh, puzzle, verbose=False):

    """
    Assume we are given an image without noise. We want to dilate the puzzle given to avoid black lines and
    get just the number.
    """

    # Make white lines thicker to get just the number, not the cell that it is bounded by.
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))  # adjust the kernel size
    dilated = cv.dilate(thresh, kernel, iterations=1)  # adjust the number of iterations

    if verbose:
        cv.imshow("Dilate puzzle", dilated)
        cv.waitKey(0)

    # Fix horizontal and vertical lines again
    vertical_kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, 3))
    vert_fixed = cv.morphologyEx(dilated, cv.MORPH_CLOSE, vertical_kernel, iterations=10)

    horizontal_kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 1))
    fixed_img = cv.morphologyEx(vert_fixed, cv.MORPH_CLOSE, horizontal_kernel, iterations=10)

    if verbose:
        cv.imshow("Again, fix horizontal an vertical just in case", fixed_img)
        cv.waitKey(0)

    # Sort by top to bottom and each row by left to right
    invert = 255 - fixed_img
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
    sudoku_cells = []
    sudoku_cells_thresh = []

    for row in sudoku_rows:
        for c in row:

            mask = np.zeros(puzzle.shape, dtype=np.uint8)
            cv.drawContours(mask, [c], -1, (255,255,255), -1)
            sudoku_cells_thresh.append(mask)
                
            result = cv.bitwise_and(puzzle, mask)
            result[mask==0] = 255
            if verbose:
                cv.imshow("Cell", result)
                cv.waitKey(0)
            sudoku_cells.append(result)
    
    return (thresh, sudoku_cells, sudoku_cells_thresh)


def process_cell(cell, cell_thresh):

    # Find white pixels
    white_pixels = np.where(cell_thresh == 255)
    if white_pixels[0].size > 0 and white_pixels[1].size > 0:
        # find 
        topmost = np.min(white_pixels[0])
        bottommost = np.max(white_pixels[0])
        leftmost = np.min(white_pixels[1])
        rightmost = np.max(white_pixels[1])

        return cell[topmost:bottommost+1, leftmost:rightmost+1] #sum one because python slicing goes to init:end-1
    
    return None


def get_number(img, NUMBER_TEMPLATES, verbose=False):

    # con umbralización del color
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    _, thresh = cv.threshold(gray, 100, 255, cv.THRESH_BINARY)
    
    if len(img.shape) == 3:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    _, img = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
    img = cv.bitwise_not(img)

    if verbose:
        cv.imshow("img threshed", img)
        cv.waitKey(0)

    if np.all(thresh == 255):
        return 0
    else:
        results = []

        # --> Get just the number in img

        # Get extreme pixels of the number
        white_pixels = np.where(img == 255)
        topmost = np.min(white_pixels[0])
        bottommost = np.max(white_pixels[0])
        leftmost = np.min(white_pixels[1])
        rightmost = np.max(white_pixels[1])

        # Get img region of interest (ROI)
        new_height = bottommost - topmost
        new_width = rightmost - leftmost
        img_ROI = img[topmost:bottommost, leftmost:rightmost]

        if verbose:
            cv.imshow("img threshed", img_ROI)
            cv.waitKey(0)

        for idx, template in enumerate(NUMBER_TEMPLATES):

            if len(template.shape) == 3:
                template = cv.cvtColor(template, cv.COLOR_BGR2GRAY)
            _, template = cv.threshold(template, 127, 255, cv.THRESH_BINARY)
            template = cv.bitwise_not(template)

            # --> Get just the number in template

            # Get extreme pixels of the template
            white_pixels = np.where(template == 255)
            topmost = np.min(white_pixels[0])
            bottommost = np.max(white_pixels[0])
            leftmost = np.min(white_pixels[1])
            rightmost = np.max(white_pixels[1])

            # Get template region of interest (ROI)
            template_ROI = template[topmost:bottommost, leftmost:rightmost]
            
            # --> Resize template to be the same size of the number
            # might work better if original aspect ratio is maintained, adjusting width to new height:
            # height, width = template_ROI.shape[:2]
            # new_width = int(width * (new_height / height))
            template_ROI = cv.resize(template_ROI, (new_width, new_height))

            if verbose:
                cv.imshow("template threshed & resized", template_ROI)
                cv.waitKey(0)

            result = cv.matchTemplate(img_ROI, template_ROI, cv.TM_CCOEFF_NORMED)

            # Find the position of the best match
            _, max_val, _, _ = cv.minMaxLoc(result)
            results.append(max_val)
        
        return results.index(max(results)) + 1

def get_sudoku(image, verbose : bool = False, detection_attempts : int = 5) -> pd.DataFrame:
    
    puzzle, thresh = find_puzzle(image, verbose=verbose)

    attempt_count = 0
    while attempt_count < detection_attempts:
        _, sudoku_cells, sudoku_cells_thresh = get_sudoku_squares(thresh, puzzle, verbose=verbose)
        attempt_count += 1

        if len(sudoku_cells) == 81:
            break

        print(f"Attempt {attempt_count}: Couldn't detect the sudoku -> {len(sudoku_cells)} cells detected")

        if attempt_count == detection_attempts:
            raise SudokuDetectionError(len(sudoku_cells), detection_attempts)
    
    cropped_cells = []
    for i in range(len(sudoku_cells)):
        processed_cell = process_cell(sudoku_cells[i], sudoku_cells_thresh[i])
        cropped_cells.append(processed_cell)

    sudoku_arr = []
    for cell in cropped_cells:

        if verbose:
            cv.imshow('cropped cell to analyze', cell)
            cv.waitKey(0)

        sudoku_arr.append(get_number(cell, NUMBER_TEMPLATES, verbose))

    return np.array(sudoku_arr).reshape(9, 9)


def main():
    image = cv.imread('res/photos/sudoku/sudokuHouse.jpg')
    cv.imshow('original image', image)
    cv.waitKey(0)

    get_sudoku(image, True, 1)

if __name__ == "__main__":
    main()