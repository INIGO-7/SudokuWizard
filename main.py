from sudoku_algorithms import *
from sudoku_identifier import SudokuWizard
import cv2 as cv
import pandas as pd

# Load the image and get the sudoku.
image = cv.imread('res/photos/sudoku/SudokuImg.jpg')
cv.imshow('Original image', image)
cv.waitKey(0)
sw = SudokuWizard(image)

sw.run(verbose=True, ocr=False)