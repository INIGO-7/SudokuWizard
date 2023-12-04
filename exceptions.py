
class SudokuDetectionError(Exception):
    """Exception raised when failing to detect all Sudoku cells."""

    def __init__(self, detected_cells, attempts):
        self.message = f"Failed to detect 81 Sudoku cells. Detected {detected_cells} cells after {attempts} attempts."
        super().__init__(self.message)
