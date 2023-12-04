from sudoku_algorithms import *
from sudoku_identifier import *
import cv2 as cv
import pandas as pd

# Load the image and get the sudoku.
image = cv.imread('res/photos/sudoku/sudoku-puzzle-games.webp')
sudoku = get_sudoku(image)

# Solve the sudoku printing original and solved on screen.
p = problemSudoku(sudoku)
print("Initial state: ")
printSudoku(p.initial_state)
res = BFS(p)
print("Solved sudoku: ")
printSudoku(res["final_state"]["state"])
