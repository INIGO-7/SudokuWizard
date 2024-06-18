from setuptools import setup, find_packages

VERSION = '0.0.1'
DESCRIPTION = 'Simple program to solve Sudoku puzzles only with the source image.'
LONG_DESCRIPTION = """Sudoku Wizard is a Python project that digitizes and solves Sudoku puzzles 
from images using advanced image processing techniques and OCR (Optical Character Recognition)."""

# Read the contents of your README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='sudokuwizard',
    version=VERSION,
    author='G.Guillotine - (INIGO-7 on GitHub)',
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',  # This ensures the README.md file is parsed correctly
    url='https://github.com/INIGO-7/SudokuWizard',  # Add your GitHub repository here
    packages=find_packages(where="SudokuWizard"),
    package_dir={"": "SudokuWizard"},
    install_requires=[
        'opencv-python',
        'pandas',
        'numpy',
        'imutils',
        'scikit-image',
        'easyocr',
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    python_requires='>=3.6',
)