# Sudoku Wizard 💻🪄

Sudoku Wizard is a Python project that digitizes and solves Sudoku puzzles from images using advanced image processing techniques and OCR (Optical Character Recognition).

![Screenshot from 2024-06-18 14-46-34](https://github.com/INIGO-7/SudokuSolver/assets/58185185/b4ae52d5-a8a4-4fe2-9d82-e30dd35ab044)

## Features

- **Image Processing**: Detects and extracts Sudoku grids from images.
- **OCR and Template Matching**: Recognizes digits within the Sudoku grid using either OCR or template matching.
- **Sudoku Solving**: Solves the extracted Sudoku puzzle using a DFS algorithm.
- **Visualization**: Overlays the solution onto the original image.

## Installation

Requires Python 3.10 or newer, on Windows, macOS or Linux.

```bash
pip install sudokuwizard          # template matching (light, ~0.5 GB with dependencies)
pip install "sudokuwizard[ocr]"   # also installs EasyOCR + PyTorch for --ocr mode (several GB)
```

### From source

```bash
git clone https://github.com/INIGO-7/SudokuWizard.git
cd SudokuWizard
pip install -e ".[ocr]"
```

## Command line

```bash
sudokuwizard photo.jpg                 # solve and show the result in a window
sudokuwizard photo.jpg -o solved.png   # also save the solved image
sudokuwizard photo.jpg --no-show -o solved.png   # servers / no display
```

## Usage

Here's a quick example of how to use Sudoku Wizard:

```python
from SudokuWizard import SudokuWizard

# Create a SudokuWizard instance
sw = SudokuWizard()

# Load your sudoku image (works with .jpg, .jpeg, .png and .webp; support for other extensions is unknown)
sw.load_image('image.jpg')

# Run the Sudoku digitalization and resolution process.
# Shows the solution in a window (pass show=False to skip) and returns the solved image.
solved = sw.run()
```

## Project Structure

- `sudoku_wizard.py`: Main implementation of the Sudoku Wizard class.
- `sudoku_algorithms.py`: Contains various algorithms for solving Sudoku puzzles.
- `exceptions.py`: Custom exceptions for error handling.
- `res/photos/`: Contains template images for number recognition.
- `pyproject.toml`: Packaging metadata and dependencies.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request on GitHub.

## License

This project is licensed under the MIT License.

---

Feel free to reach out if you have any questions or need further assistance!

---

**Enjoy solving Sudoku puzzles with Sudoku Wizard!**

# 🪄

