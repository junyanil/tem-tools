# TEM Tools



## Features

`saed_indexer.py`

- open SAED images from a full file path
- inspect azimuthal intensity profiles
- calibrate reciprocal-space scale
- pick diffraction rings or spots
- export annotated SVG files with embedded metadata

`hrtem_analyser.py`

- open HRTEM images from a full file path
- select a square ROI and inspect its FFT
- calibrate real-space pixel size
- measure diffraction spots in reciprocal space
- export combined SVG reports with embedded metadata

## Supported formats

- `dm3`
- `dm4`
- `emd`
- `emi`
- `tif`, `tiff`
- `jpg`, `jpeg`, `png`
- `mrc`

Notes:

- `dm3`, `dm4`, `emd`, `emi`, `tif`, and `tiff` are read through `rosettasciio`
- plain image formats can still be opened, but scale usually needs manual calibration

## Installation

```bash
pip install -r requirements.txt
```

## Run

```bash
python TEM_Tools/saed_indexer.py
python TEM_Tools/hrtem_analyser.py
```
