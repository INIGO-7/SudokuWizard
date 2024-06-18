from sudoku_algorithms import *
from sudoku_identifier import SudokuWizard
import cv2 as cv
import pandas as pd
import time
import os
import re

res_path = 'res/photos/sudoku'

def solve_all():
    for path in os.listdir(res_path):
        if not re.search(r'\.(jpg|jpeg|png|webp)$', path):
            continue

        sw = SudokuWizard()
        sw.load_image(os.path.join(res_path, path))
        sw.run(verbose=False, ocr=True)


# Load the image and get the sudoku.
img_path = os.path.join(res_path, 'sudokuLibro3.jpeg')

sw = SudokuWizard()
sw.load_image(img_path)
sw.run(verbose=False, ocr=False)
