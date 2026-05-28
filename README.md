# Crack Detection Toolset

An interactive GUI toolset for detecting and measuring cracks and pores in microscopy images. Users step through a guided pipeline to preprocess the image, define a region of interest, detect defects, manually refine contours, calibrate a physical scale, and export quantitative results.

## Requirements

- Python 3.13+
- PyQt5 (`pip install PyQt5`)
- All other dependencies are listed in `pyproject.toml`

Install dependencies:

```bash
pip install PyQt5
pip install -e .
```

The edge detection model (`model.yml.gz`) must be present in the working directory when running from source.

## Usage

```bash
python main.py
```

The tool guides you through the following steps via interactive GUI windows:

1. **Input** — Select an input image (`.jpg`, `.png`, or `.tiff`) and an output path.
2. **Preprocessing** — Adjust CLAHE contrast enhancement and Gaussian blur parameters.
3. **Region of interest** — Draw a polygon on the image to define the analysis area (left-click to add points, `C` to close, `R` to reset).
4. **Crack detection** — Tune confidence threshold and morphological expansion to control which edges are detected.
5. **Contour editing** — Manually add, delete, move, scale, or rotate contours. Set the circularity threshold to classify contours as cracks (low circularity, shown in green) or pores (high circularity, shown in blue).
6. **Scale calibration** — Click two points on a scale bar in the image and enter its real-world length in millimeters.
7. **Results** — A summary of crack/pore areas, area fractions, and skeleton length is displayed. The annotated image is saved to the specified output path.

## Output

The saved image shows:
- **Green** contours — cracks
- **Blue** contours — pores
- **Red** overlay — crack skeleton (centerline)

A statistics popup reports areas (µm²), area fractions relative to the region of interest, and skeleton length (µm).

## Building a standalone executable

```bash
python compiler.py
```

Requires `molecule.ico` and `model.yml.gz` in the project root. Produces a Windows executable in `dist/`.
