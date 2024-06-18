
class CellDetectionError(Exception):
    """Exception raised when failing to identify all Sudoku cells."""

    def __init__(self, detected_cells):
        self.message = f"Failed to identify correctly the Sudoku cells. Detected {detected_cells} cells (should be 81), please retake the image and try again."
        super().__init__(self.message)

class SudokuSolutionError(Exception):
    """Exception raised when the sudoku numbers haven't been detected correctly, or when there is no solution"""

    def __init__(self):
        self.message = "The sudoku couldn't be correctly solved. This might be because some number wasn't detected correctly, or because there is no solution."\
        " Please retake the image and try again. If various images return the same result, it is likely this sudoku doesn't have a solution."
        super().__init__(self.message)