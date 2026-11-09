"""Command-line entry point: `sudokuwizard photo.jpg` or `python -m SudokuWizard photo.jpg`."""
import argparse

import cv2 as cv

from .sudoku_wizard import SudokuWizard


def main(argv=None):
    parser = argparse.ArgumentParser(prog="sudokuwizard", description="Solve a sudoku from a photo.")
    parser.add_argument("image", help="Path to a photo of a sudoku (.jpg, .png, .webp...)")
    parser.add_argument("-o", "--output", help="Save the solved image to this path")
    parser.add_argument("--ocr", action="store_true", help="Use EasyOCR (needs: pip install 'sudokuwizard[ocr]')")
    parser.add_argument("--gpu", action="store_true", help="Use the GPU for OCR")
    parser.add_argument("--no-show", action="store_true", help="Don't open a window with the result")
    args = parser.parse_args(argv)

    sw = SudokuWizard(use_gpu=args.gpu)
    sw.load_image(args.image)
    solved = sw.run(ocr=args.ocr, show=not args.no_show)
    print(sw.solution)
    if args.output:
        cv.imwrite(args.output, solved)
        print(f"Saved solution to {args.output}")


if __name__ == "__main__":
    main()
