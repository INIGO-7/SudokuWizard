from sudoku_algorithms import *
from sudoku_identifier import SudokuWizard
import cv2 as cv
import pandas as pd

# Load the image and get the sudoku.
image = cv.imread('res/photos/sudoku/sudokuLibro1.jpeg')
sw = SudokuWizard(image)

sw.run()