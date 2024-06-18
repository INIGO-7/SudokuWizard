from sudoku_algorithms import *
from sudoku_identifier import SudokuWizard
import cv2 as cv
import pandas as pd
import time
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

picture = cv.imread(os.path.join(img_path, 'sudokuLibro1.jpeg'))

# First call without OCR
sw = SudokuWizard(picture)
start_time = time.time()
sw.run(verbose=False, ocr=False)
end_time = time.time()
time_without_ocr = end_time - start_time
print(f"Time taken for sw.run with OCR=False: {time_without_ocr:.4f} seconds")

# Second call with OCR
# sw = SudokuWizard(picture, True)
# start_time = time.time()
# sw.run(verbose=False, ocr=True)
# end_time = time.time()
# time_with_ocr = end_time - start_time
# print(f"Time taken for sw.run with OCR=True: {time_with_ocr:.4f} seconds")
