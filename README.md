# Sudoku Wizard 💻🪄

Sudoku Wizard is a Python project that digitizes and solves Sudoku puzzles from images using advanced image processing techniques and OCR (Optical Character Recognition).

![Screenshot from 2024-06-18 14-46-34](https://github.com/INIGO-7/SudokuSolver/assets/58185185/b4ae52d5-a8a4-4fe2-9d82-e30dd35ab044)

## Features

- **Image Processing**: Detects and extracts Sudoku grids from images.
- **OCR and Template Matching**: Recognizes digits within the Sudoku grid using either OCR or template matching.
- **Sudoku Solving**: Solves the extracted Sudoku puzzle using a DFS algorithm.
- **Visualization**: Overlays the solution onto the original image.

## Installation

To install Sudoku Wizard, follow these steps:

### Prerequisites

- Python 3.8 or later
- Git

### Clone the Repository

```bash
git clone https://github.com/yourusername/SudokuWizard.git
cd SudokuWizard
```

### Using `requirements.txt`

To install the dependencies using `pip`, run:

```bash
pip install -r requirements.txt
```

### Using `environment.yml`

To create a conda environment and install the dependencies, run:

```bash
conda env create -f environment.yml
conda activate sudoku-wizard
```

## Usage

Here's a quick example of how to use Sudoku Wizard:

```python
import cv2 as cv
from sudoku_wizard import SudokuWizard

# Create a SudokuWizard instance
sw = SudokuWizard()

# Load your sudoku image (works with .jpg, .jpeg, .png and .webp; support for other extensions is unknown)
sw.load_image('image.jpg')

# Run the Sudoku digitalization and resolution process
sw.run()
```

## Project Structure

- `sudoku_wizard.py`: Main implementation of the Sudoku Wizard class.
- `sudoku_algorithms.py`: Contains various algorithms for solving Sudoku puzzles.
- `exceptions.py`: Custom exceptions for error handling.
- `res/photos/`: Contains template images for number recognition.
- `requirements.txt`: Python dependencies.
- `environment.yml`: Conda environment configuration.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request on GitHub.

## License

This project is licensed under the MIT License.

---

Feel free to reach out if you have any questions or need further assistance!

---

Enjoy solving Sudoku puzzles with Sudoku Wizard! 🧩✨

