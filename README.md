# TEM Tools

Minimal GitHub release of two TEM analysis GUIs for small-group sharing:

- `TEM_Tools/saed_indexer.py`
- `TEM_Tools/hrtem_analyser.py`

This release was simplified for easier reuse:

- only the two main GUI files are kept
- each file contains its own minimal hyperspy/`rosettasciio` loader
- data import uses a single full file path input
- SVG export and SVG-based session restore are preserved

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

## Suggested checks before publishing

- confirm both GUIs start correctly
- open at least one `dm4` or `emd` file in each tool
- export an SVG from each tool
- reopen the exported SVG to verify metadata restore
- make sure no raw experimental data is included in the repository

## Publish to GitHub

```bash
cd "/Users/junyan/Library/CloudStorage/OneDrive-个人/python/tem_tools_github"
git init
git add .
git commit -m "Initial release of TEM tools"
git branch -M main
git remote add origin https://github.com/<your-name>/tem-tools.git
git push -u origin main
```
