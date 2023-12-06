
class SudokuDetectionError(Exception):
    """Exception raised when failing to detect all Sudoku cells."""

    def __init__(self, detected_cells):
        self.message = f"Failed to detect all the Sudoku cells. Detected {detected_cells} cells (should be 81), please retake the image and try again."
        super().__init__(self.message)
