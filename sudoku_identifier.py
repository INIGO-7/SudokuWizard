import cv2 as cv
import pandas as pd
import numpy as np
import threading
import os
import re
import sys
import time

from typing import List
from exceptions import CellDetectionError, SudokuSolutionError

import imutils
from imutils.perspective import four_point_transform
from imutils import contours
from skimage.segmentation import clear_border
import easyocr

from sudoku_algorithms import BFS, problemSudoku, printSudoku


class SudokuWizard():

    """
    A class for detecting and extracting Sudoku puzzles from images.

    This class uses image processing techniques to detect a Sudoku grid, extract each cell, 
    identify the numbers present, and return a 2D array representing the Sudoku puzzle.

    Init parameters
    ---------------

        image : np.ndarray - The original sudoku image, in opencv format.
        use_gpu : bool = False - Set to True to make some operations faster using a GPU instead of a CPU.

    Attributes
    ----------

        TEMPLATES : List[np.ndarray] - A list containing nine number templates (1-9) for number recognition.
        image : np.ndarray - The input image containing the Sudoku puzzle.
        sudoku : np.ndarray - The cropped original image with just the Sudoku puzzle.
        sudoku_thresh : np.ndarray - Thresholded version of the sudoku only with the grid, for cell extraction.
        cells : List[np.ndarray] - List of images of individual cells in the Sudoku grid.
        cells_bounding_box : 
        sudoku_arr : np.array - Array representing the numbers in the Sudoku grid.
        solution
        font
        font_color
        ocr_reader
        use_gpu
    """

    def __init__(self, use_gpu : bool = False):

        # Get the number templates for template matching.
        self.TEMPLATES: List[np.ndarray] = [
            cv.imread(f'res/photos/numbers/number{i}HQ_nomargin.jpg') 
            for i in range(1, 10)
        ]

        # Error handling in case any template isn't found, we have to stop.
        for idx, img in enumerate(self.TEMPLATES):
            if img is None:
                raise FileNotFoundError(f"Number template {idx + 1} not found or failed to load.")

        self.sudoku = None
        self.sudoku_thresh = None
        self.cells = []
        self.cells_bounding_box = []
        self.sudoku_arr = []
        self.solution = []

        self.font = cv.FONT_HERSHEY_SIMPLEX
        self.font_color = (8, 8, 161)  # Paint color

        self.ocr_reader = None
        self.use_gpu = use_gpu

        # Flag to control the loading animation
        self.stop_animation = False

    def load_image(self, img_path : str):

        """
        Load an image containing a sudoku for detection

        Args:
            img_path (str): System path where the image to read is located
        """

        self.image = cv.imread(img_path)

        if self.image is None:
            raise FileNotFoundError("Source image not found or failed to load.")


    def animate_message(self, message : str, animation : List = None):

        """
        Displays an animation in the terminal with the given message.

        Args:
            message (str): The base message to display alongside the spinner.
            animation (List): Animation to perform after the message.

        The spinner runs until the global flag `stop_animation` is set to `True`.
        """

        # Define the animation pattern
        if not animation:
            animation = ["|", "/", "-", "\\"]

        for i in range(len(message) + 1):
                if self.stop_animation:
                    break
                # Display the message letter by letter
                sys.stdout.write(f"\r{message[:i]}")
                sys.stdout.flush()
                time.sleep(0.1)  # Adjust the speed of the letter-by-letter effect here
    
        while not self.stop_animation:
            for frame in animation:
                if self.stop_animation:
                    break
                # Print the message with the current frame of the spinner
                sys.stdout.write(f"\r{message} {frame}")
                sys.stdout.flush()
                time.sleep(0.1)  # Adjust the speed of the spinner animation here
        
        # Clean up the line after stopping the animation
        sys.stdout.write("\rFinished.          \n")
        sys.stdout.flush()


    def resize_if_large(self, image : np.ndarray, max_width : int, max_height : int, scale_factor : int = 0.9):

        """
        Resizes the image if it's larger than the specified maximum width and height.

        Args:
            image (np.ndarray): The input image to resize.
            max_width (int): The maximum allowed width of the image.
            max_height (int): The maximum allowed height of the image.
            scale_factor (float): The factor by which to scale the image down each iteration.

        Returns:
            np.ndarray: The resized image.
        """
        
        # Get image dimensions
        height, width = image.shape[:2]

        while(width > max_width or height > max_height):
            # Calculate new dimensions
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)

            # Resize the image
            image = cv.resize(image, (new_width, new_height), interpolation=cv.INTER_AREA)
            height, width = image.shape[:2]

        return image

    def scan_image(self, verbose : bool = False) -> (np.ndarray, np.ndarray):

        """
        Scans the input image to find the Sudoku grid and preprocesses it.

        This method converts the image to grayscale, applies adaptive thresholding, and uses morphological transformations to highlight the grid. The method returns the processed image and a thresholded version of it.

        Args:
            verbose (bool): If True, displays intermediate steps for debugging.

        Returns:
            tuple: A tuple containing the processed Sudoku image and its thresholded version.
        """

        # Resize the image if it's too large
        self.image = self.resize_if_large(self.image, 1920, 1080)

        # Convert the image to grayscale, then apply a threshold to get black and white details.
        gray = cv.cvtColor(self.image, cv.COLOR_BGR2GRAY)
        thresh = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 57, 12)

        if verbose:
            cv.imshow("Black and white image with adaptive threshold", thresh)
            cv.waitKey(0)

        # Make white lines thicker to identify them better
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (2, 2))  # adjust the kernel size
        dilated = cv.dilate(thresh, kernel, iterations=1)  # adjust the number of iterations

        if verbose:
            cv.imshow("Slight dilation of lines", dilated)
            cv.waitKey(0)

        # Morphological closing to connect lines (reduce gaps in lines)
        closing_kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
        closed = cv.morphologyEx(dilated, cv.MORPH_CLOSE, closing_kernel)

        if verbose:
            cv.imshow("Close gaps between lines", closed)
            cv.waitKey(0)

        # Blur to reduce noise caused by dilating the content of the image (which could make noise more present)
        blurred = cv.medianBlur(closed, 3)

        if verbose:
            cv.imshow("Remove noise created so far", blurred)
            cv.waitKey(0)

        # Filter out all numbers and noise to isolate only boxes
        cnts = cv.findContours(blurred, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        for c in cnts:
            area = cv.contourArea(c)
            if area < 1200:
                cv.drawContours(blurred, [c], -1, (0,0,0), -1)

        if verbose:
            cv.imshow("Keep identified contours", blurred)
            cv.waitKey(0)

        # Morphological closing again to try to connect the lines
        closing_kernel = cv.getStructuringElement(cv.MORPH_RECT, (9, 9))
        closed = cv.morphologyEx(blurred, cv.MORPH_CLOSE, closing_kernel)

        if verbose:
            cv.imshow("Close gaps between lines, now more aggressive", closed)
            cv.waitKey(0)

        # Fix vertical and horizontal lines to be more clear and well defined
        vertical_kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, 3))
        fixed_img = cv.morphologyEx(closed, cv.MORPH_CLOSE, vertical_kernel, iterations=10)

        horizontal_kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 1))
        fixed_img = cv.morphologyEx(fixed_img, cv.MORPH_CLOSE, horizontal_kernel, iterations=10)

        if verbose:
            cv.imshow("Process horizontal and vertical lines", fixed_img)
            cv.waitKey(0)

        # find contours in the thresholded image and sort them by size in descending order
        cnts = cv.findContours(fixed_img.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = sorted(cnts, key=cv.contourArea, reverse=True)

        # initialize sudoku_contour, which will correspond to the sudoku outline
        sudoku_contour = None

        # loop over the contours
        for c in cnts:

            # approximate the contour
            peri = cv.arcLength(c, True)
            approx = cv.approxPolyDP(c, 0.02 * peri, True)
            # if our approximated contour has four points, then we can assume we have found the outline of the sudoku
            if len(approx) == 4:
                sudoku_contour = approx
                break

        # if sudoku_contour is empty -> script could not find the outline of the Sudoku sudoku so raise an error
        if sudoku_contour is None:
            raise Exception(("Could not find Sudoku sudoku outline. "
            "Try debugging your thresholding and contour steps."))
        
        # if we're in verbose mode -> check to see if the obtained outline of the Sudoku sudoku is correct 
        if verbose:
            outline = self.image.copy()
            cv.drawContours(outline, [sudoku_contour], -1, (0, 255, 0), 2)
            cv.imshow("Sudoku outline", outline)
            cv.waitKey(0)

        # apply a four point perspective transform to both the original image and black/white image to obtain a
        # top-down bird's eye view of the sudoku
        sudoku = four_point_transform(self.image, sudoku_contour.reshape(4, 2))
        sudoku_thresh = four_point_transform(fixed_img, sudoku_contour.reshape(4, 2))

        if verbose:
            cv.imshow("Identified sudoku", sudoku)
            cv.waitKey(0)

        if verbose:
            cv.imshow("Identified cells", sudoku_thresh)
            cv.waitKey(0)
        
        self.sudoku = sudoku
        self.sudoku_thresh = sudoku_thresh

        # return a 2-tuple of the cropped sudoku in both RGB and B/W
        return (sudoku, sudoku_thresh)

    def crop_by_contour(self, image, contour):

        """
        This function returns an image that is cropped with a contour given by parameter.

        Args:
            image (np.ndarray): The input image to crop.
            contour (np.ndarray): The contour used for cropping.

        Returns:
            np.ndarray: The cropped image.
        """

        # Find white pixels (the contour)
        white_pixels = np.where(contour == 255)

        # Only if there are more than 16 white pixels in the image (we have at least the contour of a small cell)
        if white_pixels[0].size > 4 and white_pixels[1].size > 4:
            
            topmost = np.min(white_pixels[0])
            bottommost = np.max(white_pixels[0])
            leftmost = np.min(white_pixels[1])
            rightmost = np.max(white_pixels[1])

            return image[topmost:bottommost+1, leftmost:rightmost+1] #sum one because python slicing goes to init:end-1
        
        # If we don't have a contour, then we end the process
        else:
            raise Exception(("The contour of this cell is invalid!"))

    def extract_cells(self, verbose : bool = False):

        """
        Assume we already have an image of the cropped sudoku without noise (func scan_sudoku must run first).
        We will then store an array of images containing the cropped sudoku cells, for further processing.

        Args:
            verbose (bool): If True, displays each cell during extraction.

        Returns:
            tuple: A tuple containing arrays of cell images and their thresholded versions.
        """
        

        if self.sudoku_thresh is None:
            raise Exception(("function scan_image must be run first to extract the sudoku!!"))

        # Make white lines thicker to get the number without the cell that it is bounded by.
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))  # adjust the kernel size
        dilated = cv.dilate(self.sudoku_thresh, kernel, iterations=3)  # adjust the number of iterations

        if verbose:
            cv.imshow("Sudoku with dilated lines", dilated)
            cv.waitKey(0)

        # Fix horizontal and vertical lines again, very important to reduce noise!
        vertical_kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, 3))
        vert_fixed = cv.morphologyEx(dilated, cv.MORPH_CLOSE, vertical_kernel, iterations=10)

        horizontal_kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 1))
        fixed_img = cv.morphologyEx(vert_fixed, cv.MORPH_CLOSE, horizontal_kernel, iterations=10)

        if verbose:
            cv.imshow("Horizontal and vertical lines corrected preventively", fixed_img)
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

                # Once 9 cells have been added to the row, they are sorted and added to sudoku_rows
                if i % 9 == 0:  
                    (cnts, _) = contours.sort_contours(row, method="left-to-right")
                    sudoku_rows.append(cnts)
                    row = []
                    

        for row in sudoku_rows:
            for cell_contour in row:

                # A crop done with the bounding rect of each contour is not good enough, we need
                # the precision the contour provides, and this is in jeopardy when we get a 
                # rectangle from the contour. So here we get the exact crop of the original image
                # where a contour has been detected.

                self.cells_bounding_box.append(cv.boundingRect(cell_contour))

                mask = np.zeros(self.sudoku.shape, dtype=np.uint8)
                cv.drawContours(mask, [cell_contour], -1, (255,255,255), -1)

                highlighted_cell = cv.bitwise_and(self.sudoku, mask)
                highlighted_cell[mask==0] = 255

                cropped_cell = self.crop_by_contour(highlighted_cell, mask)
                self.cells.append(cropped_cell)

                if verbose:
                    cv.imshow("Cropped cell", cropped_cell)
                    cv.waitKey(150)

        if len(self.cells) != 81:
            raise CellDetectionError(len(self.cells))

        return self.cells


    def get_cell_number(self, cell : np.ndarray, verbose : bool = False, ocr : bool = False):

        """
        Extracts the number from a cell using either OCR or template matching.

        Args:
            cell (np.ndarray): The input cell image.
            verbose (bool): If True, displays intermediate steps for debugging.
            ocr (bool): If True, uses OCR for number recognition; otherwise uses template matching.

        Returns:
            int: The recognized number in the cell, or 0 if the cell is empty.
        """
        
        # Threshold to show the number, or nothing if there is not a number in the cell
        gray = cv.cvtColor(cell, cv.COLOR_BGR2GRAY)
        _, thresh = cv.threshold(gray, 100, 255, cv.THRESH_BINARY)

        if verbose:
            cv.imshow("Is there a number here?", cell)
            cv.waitKey(0)

        # If there wasn't any number (no white in cell), we return 0 (which represents an empty cell)
        if np.all(thresh == 255):
            return 0
        
        # If there was something in the cell, use ocr or template matching to determine what number is it.
        elif ocr:
            
            result = self.ocr_reader.recognize(cell)[0][1]
            result = re.search(r'\d', result).group()
            return int(result)

        else:

            if len(cell.shape) == 3:
                cell = cv.cvtColor(cell, cv.COLOR_BGR2GRAY)
            
            #Threshold to get the number as clear as possible
            _, cell = cv.threshold(cell, 127, 255, cv.THRESH_BINARY)
            cell = cv.bitwise_not(cell)
            results = []

            # --> Get just the number in cell

            # Get extreme pixels of the number
            white_pixels = np.where(cell == 255)
            topmost = np.min(white_pixels[0])
            bottommost = np.max(white_pixels[0])
            leftmost = np.min(white_pixels[1])
            rightmost = np.max(white_pixels[1])

            # Get the cell's region of interest (ROI), which is the number
            new_height = bottommost - topmost
            new_width = rightmost - leftmost
            cell_ROI = cell[topmost:bottommost, leftmost:rightmost]

            if verbose:
                cv.imshow("Yes, get our cell's ROI", cell_ROI)
                cv.waitKey(0)
            
            for template in self.TEMPLATES:

                # We turn the image into grayscale if we have a coloured image
                if len(template.shape) == 3:
                    template = cv.cvtColor(template, cv.COLOR_BGR2GRAY)

                # The same threshold applied to the cell is applied to the template 
                _, template = cv.threshold(template, 127, 255, cv.THRESH_BINARY)
                template = cv.bitwise_not(template)

                # --> Get just the number in template

                # Get extreme pixels of the template
                white_pixels = np.where(template == 255)
                topmost = np.min(white_pixels[0])
                bottommost = np.max(white_pixels[0])
                leftmost = np.min(white_pixels[1])
                rightmost = np.max(white_pixels[1])

                # Get the template's region of interest (ROI), which is the number
                template_ROI = template[topmost:bottommost, leftmost:rightmost]
                
                # --> Resize template to be the same size of the number, without maintaining original aspect ratio.
                # This gives a much better result than maintaining the aspect ratio.
                template_ROI = cv.resize(template_ROI, (new_width, new_height))

                if verbose:
                    cv.imshow("Our template's ROI", template_ROI)
                    cv.waitKey(0)

                result = cv.matchTemplate(cell_ROI, template_ROI, cv.TM_CCOEFF_NORMED)

                # Find the position of the best match
                _, max_val, _, _ = cv.minMaxLoc(result)
                results.append(max_val)
                

            # Return the index+1 with the highest match ratio, which would be the template it resembles the most to.
            return results.index(max(results)) + 1

    def extract_numbers(self, verbose : bool = False, ocr : bool = False) -> np.array:

        """
        Extracts numbers from each cell in the Sudoku grid using either OCR or template matching.

        Args:
            verbose (bool): If True, displays intermediate steps for debugging.
            ocr (bool): If True, uses OCR for number recognition; otherwise uses template matching.

        Returns:
            np.array: A 2D array representing the numbers in the Sudoku grid.
        """
        
        if ocr:
            #Load our ocr reader
            self.ocr_reader = easyocr.Reader(['en'], gpu=self.use_gpu)
            
        for cell in self.cells:

            number = self.get_cell_number(cell, verbose=verbose, ocr=ocr)
            if verbose: print(f'The number in this cell is: {number}')

            self.sudoku_arr.append(number)

        self.sudoku_arr = np.array(self.sudoku_arr).reshape(9, 9)
        
        if verbose: print(self.sudoku_arr)

        return self.sudoku_arr

    def get_sudoku(self, verbose : bool = False, ocr : bool = False) -> pd.DataFrame:

        """
        Retrieves the Sudoku grid from the input image, extracts cells, and identifies numbers.

        Args:
            verbose (bool): If True, displays intermediate steps for debugging.
            ocr (bool): If True, uses OCR for number recognition; otherwise uses template matching.

        Returns:
            pd.DataFrame: A DataFrame representing the Sudoku grid.
        """
        
        # Scan the image to get a cropped image with the sudoku
        self.scan_image(verbose=verbose)

        # Get each cell from that cropped sudoku
        self.extract_cells(verbose=verbose)

        # Get all the numbers corresponding each cell, and return
        return self.extract_numbers(verbose=verbose, ocr=ocr)
    
    def solve(self, verbose : bool = False):

        """
        Solves the Sudoku puzzle using a BFS algorithm.

        Args:
            verbose (bool): If True, displays the initial and solved Sudoku states for debugging.
        """
        
        sudoku_df = pd.DataFrame(self.sudoku_arr)
        p = problemSudoku(sudoku_df)

        if verbose:
            print("Initial state: ")
            printSudoku(p.initial_state)
        
        sudoku_solved_df = BFS(p)["final_state"]["state"]

        if verbose:
            print("Solved sudoku: ")
            printSudoku(sudoku_solved_df)
        
        self.solution = sudoku_solved_df.values.flatten().reshape(9, 9)
    
    def get_font_size(self, width, height):

        """
        Calculates the font size and thickness based on cell dimensions.

        Args:
            width (int): The width of the cell.
            height (int): The height of the cell.

        Returns:
            tuple: A tuple containing the font scale and thickness.
        """
        
        # Base font scale
        base_font_scale = 1.0

        # Scale factor
        scale_factor = min(width, height) / 35.0  # Assuming 35 is approx. base cell size

        # Adjusted font size
        font_scale = base_font_scale * scale_factor

        # Estimate font thickness
        font_thickness = max(2, int(font_scale / 2))

        return font_scale, font_thickness

    def show_solution(self):

        """
        Overlays the solved Sudoku numbers on the original Sudoku image.

        Returns:
            np.ndarray: The image with the solved Sudoku overlay.
        """

        solution_img = self.sudoku.copy()

        if np.any(self.solution == 0):
            raise SudokuSolutionError()

        for row_idx in range(9):
            for cell_idx in range(9):
                if self.sudoku_arr[row_idx][cell_idx] == 0:
                    
                    number_to_write = str(self.solution[row_idx][cell_idx])
                    x, y, width, height = self.cells_bounding_box[(row_idx * 9) + cell_idx]

                    # Adjust the font size to fit the cell's dimesions
                    font_scale, font_thickness = self.get_font_size(width, height)

                    # Position adjustment to center the text in the cell
                    text_size = cv.getTextSize(number_to_write, self.font, font_scale, font_thickness)[0]
                    text_x = x + (width - text_size[0]) // 2
                    text_y = y + (height + text_size[1]) // 2

                    cv.putText(
                        solution_img, 
                        number_to_write, 
                        (text_x, text_y), 
                        self.font, 
                        font_scale, 
                        self.font_color, 
                        font_thickness
                    )
                    
        cv.destroyAllWindows()
        cv.imshow('Original sudoku', self.image)
        cv.imshow('Sudoku Solved', solution_img)
        cv.waitKey(0)
        return solution_img
    
    def run(self, verbose : bool = False, ocr : bool = False):

        """
        Executes the entire process of solving the Sudoku puzzle from an image.

        Args:
            verbose (bool): If True, displays intermediate steps for debugging.
            ocr (bool): If True, uses OCR for number recognition; otherwise uses template matching.

        Returns:
            np.ndarray: The image with the solved Sudoku overlay.
        """

        self.stop_animation = False

        # Start the animation in a separate thread
        animation_thread = threading.Thread(target=self.animate_message, args=("--> Processing",))
        animation_thread.start()

        start_time = time.time()

        try:
            # Get the sudoku from the original image
            self.get_sudoku(verbose=verbose, ocr=ocr)

            # Solve the obtained sudoku
            self.solve(verbose=verbose)

        finally:
            # Stop the animation
            self.stop_animation = True
            # Ensure the animation thread has finished
            animation_thread.join()

            end_time = time.time()
            time_without_ocr = end_time - start_time

            if ocr:
                gpu_or_not = "using GPU" if self.use_gpu else "not using GPU"
                print(f"Time taken for sw.run with artificial intelligence OCR {gpu_or_not}: {time_without_ocr:.4f} seconds")    
            else:
                print(f"Time taken for sw.run with template matching OCR: {time_without_ocr:.4f} seconds")

        # Show the solution we have found and return it
        return self.show_solution()


def main():
    image = cv.imread('res/photos/sudoku/sudokuLibroSolved1.jpeg')

    sw = SudokuWizard(image)
    sw.scan_image(verbose=True)
    sw.extract_cells()
    sw.extract_numbers(ocr=True, verbose=False)
    sw.solve()
    sw.show_solution()

if __name__ == "__main__":
    main()
