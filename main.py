from sudoku_algorithms import *
from sudoku_identifier import *
import cv2 as cv
import pandas as pd

# Load the image and get the sudoku.
image = cv.imread('res/photos/sudoku/sudokuHouse.jpg')
sudoku_arr = get_sudoku(image)
sudoku_df = pd.DataFrame(sudoku_arr)

# Solve the sudoku printing original and solved on screen.
p = problemSudoku(sudoku_df)
print("Initial state: ")
printSudoku(p.initial_state)
sudoku_solved_df = BFS(p)["final_state"]["state"]
print("Solved sudoku: ")
printSudoku(sudoku_solved_df)

sudoku_solved_arr = sudoku_solved_df.values.flatten()

# Now fill the original sudoku with the missing numbers

