import cv2 as cv
import pandas as pd
import numpy as np
import os

from typing import List
from exceptions import SudokuDetectionError

import imutils
from imutils.perspective import four_point_transform
from imutils import contours
from skimage.segmentation import clear_border


class SudokuWizard():

    """
    A class for detecting and extracting Sudoku puzzles from images.

    This class uses image processing techniques to detect a Sudoku grid, extract each cell, 
    identify the numbers present, and return a 2D array representing the Sudoku puzzle.

    Attributes
    ----------

        TEMPLATES (List[np.ndarray]): A list containing nine number templates (1-9) for number recognition.
        image (np.ndarray): The input image containing the Sudoku puzzle.
        sudoku (np.ndarray): The processed image of the Sudoku grid.
        sudoku_thresh (np.ndarray): Thresholded version of the Sudoku grid for cell extraction.
        cell_arr (List[np.ndarray]): List of images of individual cells in the Sudoku grid.
        cell_thresh_arr (List[np.ndarray]): List of thresholded images of individual cells.
        cropped_cells (List[np.ndarray]): List of cropped cell images.
        sudoku_arr (np.array): Array representing the numbers in the Sudoku grid.
    """

    def __init__(self, image : np.ndarray):

        # Get the number templates for template matching.
        self.TEMPLATES: List[np.ndarray] = [
            cv.imread(f'res/photos/numbers/number{i}HQ_nomargin.jpg') 
            for i in range(1, 10)
        ]

        # Error handling in case any template isn't found, we have to stop.
        for idx, img in enumerate(self.TEMPLATES):
            if img is None:
                raise FileNotFoundError(f"Number template {idx + 1} not found or failed to load.")

        self.image = image
        self.sudoku = None
        self.sudoku_thresh = None
        self.cells = []
        self.sudoku_arr = []


    def scan_image(self, verbose : bool = False) -> (np.ndarray, np.ndarray):

        """
        Scans the input image to find the Sudoku grid and preprocesses it.

        This method converts the image to grayscale, applies adaptive thresholding, and uses morphological transformations to highlight the grid. The method returns the processed image and a thresholded version of it.

        Args:
            verbose (bool): If True, displays intermediate steps for debugging.

        Returns:
            tuple: A tuple containing the processed Sudoku image and its thresholded version.
        """

        # Convert the image to grayscale, then apply a threshold to get black and white details.
        gray = cv.cvtColor(self.image, cv.COLOR_BGR2GRAY)
        thresh = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 57, 12)

        if verbose:
            cv.imshow("Imagen en blanco y negro con adaptative threshold", thresh)
            cv.waitKey(0)

        # Make white lines thicker to identify them better
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (2, 2))  # adjust the kernel size
        dilated = cv.dilate(thresh, kernel, iterations=1)  # adjust the number of iterations

        if verbose:
            cv.imshow("Dilatacion sutil de las lineas", dilated)
            cv.waitKey(0)

        # Morphological closing to connect lines (reduce gaps in lines)
        closing_kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
        closed = cv.morphologyEx(dilated, cv.MORPH_CLOSE, closing_kernel)

        if verbose:
            cv.imshow("Cerramos los huecos entre las lineas", closed)
            cv.waitKey(0)

        # Blur to reduce noise caused by dilating the content of the image (which could make noise more present)
        blurred = cv.medianBlur(closed, 3)

        if verbose:
            cv.imshow("Quitamos el ruido creado hasta ahora", blurred)
            cv.waitKey(0)

        # Filter out all numbers and noise to isolate only boxes
        cnts = cv.findContours(blurred, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        for c in cnts:
            area = cv.contourArea(c)
            if area < 1200:
                cv.drawContours(blurred, [c], -1, (0,0,0), -1)

        if verbose:
            cv.imshow("Nos quedamos con los contornos identificados", blurred)
            cv.waitKey(0)

        # Morphological closing again to try to connect the lines
        closing_kernel = cv.getStructuringElement(cv.MORPH_RECT, (9, 9))
        closed = cv.morphologyEx(blurred, cv.MORPH_CLOSE, closing_kernel)

        if verbose:
            cv.imshow("Cerramos huecos entre lineas, ahora mas agresivo", closed)
            cv.waitKey(0)

        # Fix vertical and horizontal lines to be more clear and well defined
        vertical_kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, 3))
        fixed_img = cv.morphologyEx(closed, cv.MORPH_CLOSE, vertical_kernel, iterations=10)

        horizontal_kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 1))
        fixed_img = cv.morphologyEx(fixed_img, cv.MORPH_CLOSE, horizontal_kernel, iterations=10)

        if verbose:
            cv.imshow("Procesamos las lineas horizontales y verticales", fixed_img)
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
            cv.imshow("Contorno del sudoku", outline)
            cv.waitKey(0)

        # apply a four point perspective transform to both the original image and black/white image to obtain a
        # top-down bird's eye view of the sudoku
        sudoku = four_point_transform(self.image, sudoku_contour.reshape(4, 2))
        sudoku_thresh = four_point_transform(fixed_img, sudoku_contour.reshape(4, 2))

        if verbose:
            cv.imshow("Este es el sudoku identificado", sudoku)
            cv.waitKey(0)

        if verbose:
            cv.imshow("Estas son las celdas identificadas", sudoku_thresh)
            cv.waitKey(0)
        
        self.sudoku = sudoku
        self.sudoku_thresh = sudoku_thresh

        # return a 2-tuple of the cropped sudoku in both RGB and B/W
        return (sudoku, sudoku_thresh)

    def crop_by_contour(self, image, contour):

        """
        This function returns an image that is cropped with a contour given by parameter.
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

    def extract_cells(self, verbose=False):

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
            cv.imshow("Sudoku con lineas dilatadas", dilated)
            cv.waitKey(0)

        # Fix horizontal and vertical lines again, very important to reduce noise!
        vertical_kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, 3))
        vert_fixed = cv.morphologyEx(dilated, cv.MORPH_CLOSE, vertical_kernel, iterations=10)

        horizontal_kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 1))
        fixed_img = cv.morphologyEx(vert_fixed, cv.MORPH_CLOSE, horizontal_kernel, iterations=10)

        if verbose:
            cv.imshow("Se corrigen las lineas horizontales y verticales preventivamente", fixed_img)
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
                    

        for row in sudoku_rows:
            for cell_contour in row:

                # A crop done with the bounding rect of each contour is not good enough, we need
                # the precision the contour provides, and this is in jeopardy when we get a 
                # rectangle from the contour. So here we get the exact crop of the original image
                # where a contour has been detected.

                mask = np.zeros(self.sudoku.shape, dtype=np.uint8)
                cv.drawContours(mask, [cell_contour], -1, (255,255,255), -1)

                highlighted_cell = cv.bitwise_and(self.sudoku, mask)
                highlighted_cell[mask==0] = 255

                cropped_cell = self.crop_by_contour(highlighted_cell, mask)
                self.cells.append(cropped_cell)

                if verbose:
                    cv.imshow("Celda recortada", cropped_cell)
                    cv.waitKey(150)

        if len(self.cells) != 81:
            raise SudokuDetectionError(len(self.cells))

        return self.cells


    def get_cell_number(self, cell : np.ndarray, verbose : bool = False):

        # Threshold to show the number, or nothing if there is not a number in the cell
        gray = cv.cvtColor(cell, cv.COLOR_BGR2GRAY)
        _, thresh = cv.threshold(gray, 100, 255, cv.THRESH_BINARY)

        if verbose:
            cv.imshow("Hay aqui un numero?", cell)
            cv.waitKey(0)

        # If there wasn't any number (no white in cell), we return 0 (which represents an empty cell)
        if np.all(thresh == 255):
            return 0
        
        # If there was something in the cell...
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
                cv.imshow("Region de interes de nuestra celda", cell_ROI)
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
                    cv.imshow("Region de interes de nuestro template", template_ROI)
                    cv.waitKey(0)

                result = cv.matchTemplate(cell_ROI, template_ROI, cv.TM_CCOEFF_NORMED)

                # Find the position of the best match
                _, max_val, _, _ = cv.minMaxLoc(result)
                results.append(max_val)
            
            # Return the index+1 with the highest match ratio, which would be the template it resembles the most to.
            return results.index(max(results)) + 1

    def extract_numbers(self, verbose : bool = False) -> np.array:

        for cell in self.cells:
            self.sudoku_arr.append(self.get_cell_number(cell, verbose))

        self.sudoku_arr = np.array(self.sudoku_arr).reshape(9, 9)

        return self.sudoku_arr

    def run(self, verbose : bool = False) -> pd.DataFrame:
        
        self.scan_image(verbose=verbose)
        self.extract_cells(verbose=verbose)
        return self.extract_numbers(verbose=verbose)


def main():
    image = cv.imread('res/photos/sudoku/sudokuHouse.jpg')

    sw = SudokuWizard(image)
    res = sw.run()

    #Show the image of the sudoku, and the obtained numbers to check if they're correct.
    print(res)
    cv.imshow('original image', image)
    cv.waitKey(0)

if __name__ == "__main__":
    main()