from sudoku_algorithms import *
from sudoku_identifier import SudokuWizard
import cv2 as cv
import pandas as pd
import os

# Load the image and get the sudoku.
img_path = 'res/photos/sudoku'

def solve_all():
    for pic in os.listdir(img_path):
        
        pic = os.path.join(img_path, pic)
        pic_cv = cv.imread(pic)
        cv.imshow('Original image', pic_cv)
        cv.waitKey(0)
        print(pic)

        sw = SudokuWizard(pic_cv)
        sw.run(verbose=False, ocr=True)

picture = cv.imread(os.path.join(img_path, 'sudokulibro4.jpeg'))
cv.imshow('Original image', picture)
cv.waitKey(0)

sw = SudokuWizard(picture)
sw.run(verbose=False, ocr=False)