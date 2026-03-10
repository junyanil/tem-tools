# hrtem_fft_analyzer.py
# Self-contained GitHub release prepared with OpenAI Codex assistance (2026-03-10).
# -*- coding: utf-8 -*-
"""
HRTEM ROI-FFT analyzer (TEM_Tools)
- Open image from a single full path
- Movable/resizable square ROI overlay (always square, clamped within image)
- FFT is computed from ROI after optional binning (none / bin2 / bin4), then fftshift centered
- Axes shown in physical units:
    * Image axes shown as nm (via AxisItem.setScale, ROI still in px coords)
    * FFT axes shown as nm^-1 (k-space), clicking picks in nm^-1
- Click diffraction spots on FFT:
    * show numbered label beside each pick
    * table lists only |k| (nm^-1) and d (nm)
- Remove selected rows updates labels and numbering
"""

import json
import re
from pathlib import Path

import numpy as np
from PyQt5 import QtCore, QtWidgets, QtGui
import pyqtgraph as pg
pg.setConfigOptions(imageAxisOrder='row-major')


def import_data_hyperspy(file_path: str):
    """Minimal rsciio loader for microscopy formats used by this GUI."""
    suffix = Path(file_path).suffix.lower()
    if suffix == ".emd":
        import rsciio.emd as rio
    elif suffix in {".dm3", ".dm4"}:
        import rsciio.digitalmicrograph as rio
    elif suffix in {".tif", ".tiff"}:
        import rsciio.tiff as rio
    elif suffix == ".emi":
        import rsciio.tia as rio
    else:
        raise ValueError(f"Unsupported hyperspy/rsciio format: {suffix}")
    return rio.file_reader(file_path)


def load_hyperspy_dataset(file_path: str | Path):
    """Load the first dataset from a hyperspy-compatible file."""
    path = Path(file_path).expanduser()
    datasets = import_data_hyperspy(path.as_posix())
    if not isinstance(datasets, (list, tuple)) or len(datasets) == 0:
        raise ValueError("No datasets were found in the selected file.")
    return datasets[0]


# -------------------------
# Utilities
# -------------------------
def to_float_image(img: np.ndarray) -> np.ndarray:
    """Convert image to float32 for processing, preserve relative contrast."""
    img = np.asarray(img)
    if img.ndim == 3:
        img = img[..., :3]
        img = 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]
    return img.astype(np.float32, copy=False)


def _to_nm_per_px(scale, units) -> float:
    """Convert (scale, units) into nm/px. Return 0.0 if unknown."""
    if scale is None:
        return 0.0
    try:
        sc = float(scale)
    except Exception:
        return 0.0
    if sc <= 0:
        return 0.0

    u = (str(units).strip().lower() if units is not None else "")
    if u in ("nm", "nanometer", "nanometers"):
        return sc
    if u in ("å", "a", "angstrom", "angstroms", "\u00c5"):
        return sc * 0.1
    if u in ("pm", "picometer", "picometers"):
        return sc * 1e-3
    if u in ("m", "meter", "meters"):
        return sc * 1e9
    return 0.0


def _clamp_roi_square(x0: float, y0: float, s: float, w: int, h: int):
    """Clamp square ROI (x0,y0,size=s) within image bounds (w,h)."""
    if w <= 1 or h <= 1:
        return 0.0, 0.0, max(1.0, s)
    s = float(max(1.0, min(s, w, h)))
    x0 = float(np.clip(x0, 0.0, max(0.0, w - s)))
    y0 = float(np.clip(y0, 0.0, max(0.0, h - s)))
    return x0, y0, s


def bin2_mean(patch: np.ndarray) -> np.ndarray:
    """2x2 binning by mean. Patch will be trimmed to even size."""
    patch = np.asarray(patch, dtype=np.float32)
    n = int(min(patch.shape[0], patch.shape[1]))
    n = (n // 2) * 2  # even
    if n < 2:
        return patch[:0, :0]
    patch = patch[:n, :n]
    return patch.reshape(n // 2, 2, n // 2, 2).mean(axis=(1, 3))


def bin4_mean(patch: np.ndarray) -> np.ndarray:
    """4x4 binning by mean. Patch will be trimmed to multiple of 4."""
    patch = np.asarray(patch, dtype=np.float32)
    n = int(min(patch.shape[0], patch.shape[1]))
    n = (n // 4) * 4
    if n < 4:
        return patch[:0, :0]
    patch = patch[:n, :n]
    return patch.reshape(n // 4, 4, n // 4, 4).mean(axis=(1, 3))


def fft_mag_log_with_axes(patch: np.ndarray, px_nm: float, bin_factor: int = 2):
    """
    Return (mag, extent_k, px_nm_eff, n)
    - mag: log magnitude FFT (fftshifted), float32
    - extent_k: (kx_min, kx_max, ky_min, ky_max) in nm^-1
    - px_nm_eff: effective pixel size after binning (nm/px)
    - n: FFT size
    """
    if bin_factor == 4:
        patch = bin4_mean(patch)
        px_nm_eff = px_nm * 4.0
    elif bin_factor == 2:
        patch = bin2_mean(patch)
        px_nm_eff = px_nm * 2.0
    else:
        patch = np.asarray(patch, dtype=np.float32)
        px_nm_eff = px_nm

    if patch.size == 0:
        return np.zeros((2, 2), np.float32), (-0.5, 0.5, -0.5, 0.5), px_nm_eff, 2

    patch = patch.astype(np.float32, copy=False)
    patch = patch - float(np.mean(patch))

    n = int(patch.shape[0])

    # window
    w = np.hanning(n).astype(np.float32)
    patch = patch * np.outer(w, w)

    F = np.fft.fftshift(np.fft.fft2(patch))
    mag = np.log1p(np.abs(F)).astype(np.float32)

    # frequency axis (cycles per nm = nm^-1)
    if px_nm_eff > 0:
        kmax = 0.5 / float(px_nm_eff)
    else:
        # fallback
        kmax = 0.5
    extent_k = (-kmax, kmax, -kmax, kmax)
    return mag, extent_k, px_nm_eff, n


# -------------------------
# Robust contrast helper
# -------------------------
def robust_levels(img: np.ndarray, p_low: float = 5.0, p_high: float = 99.8):
    """Return (lo, hi) display levels using robust percentiles to avoid over-bright background."""
    a = np.asarray(img, dtype=np.float32)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return 0.0, 1.0
    lo = float(np.percentile(a, p_low))
    hi = float(np.percentile(a, p_high))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.min(a))
        hi = float(np.max(a))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return 0.0, 1.0
    return lo, hi


def radial_profile_mean(img: np.ndarray):
    """Radial mean profile from image center. Return (r_px, mean_vals)."""
    a = np.asarray(img, dtype=np.float32)
    if a.ndim != 2 or a.size == 0:
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)

    h, w = a.shape
    cy = (h - 1) * 0.5
    cx = (w - 1) * 0.5
    yy, xx = np.indices(a.shape, dtype=np.float32)
    rr = np.hypot(xx - cx, yy - cy)
    r_int = rr.astype(np.int32)

    sums = np.bincount(r_int.ravel(), weights=a.ravel())
    cnts = np.bincount(r_int.ravel())
    valid = cnts > 0
    if not np.any(valid):
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)

    vals = np.zeros_like(sums, dtype=np.float32)
    vals[valid] = (sums[valid] / cnts[valid]).astype(np.float32)

    # keep up to Nyquist-ish radius
    r_max = int(min(h, w) // 2)
    r = np.arange(min(r_max + 1, vals.size), dtype=np.float32)
    return r, vals[:r.size]


# -------------------------
# Main Window
# -------------------------
class HRTEMFFTAnalyzer(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("HRTEM Analyzer")
        self.resize(1400, 850)

        # ---- State ----
        self.img = None
        self.img_path = None
        self.roi_size = 512  # px
        self.roi_angle_deg = 0.0
        self.points = []     # list of picks: {"kx","ky","k","d"}
        self.text_labels = []  # pg.TextItem list for numbering

        # FFT axis state
        self._fft_kmax = None
        self._fft_n = None
        self._radial_k = np.zeros(0, dtype=np.float32)
        self._radial_y = np.zeros(0, dtype=np.float32)
        self._ring_radius = None
        self._radial_vline_item = None
        self._line_profile_x = np.zeros(0, dtype=np.float32)
        self._line_profile_y = np.zeros(0, dtype=np.float32)
        self._line_profile_roi = None
        self._line_profile_box_item = None
        self._profile_mode_image = False
        self._scale_source = "none"
        self._calibrating_scale = False
        self._calib_line_roi = None
        self._updating_calib_line = False
        self._calib_bar_h_px = 4.0

        # ---- UI ----
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main_layout = QtWidgets.QHBoxLayout(central)
        main_layout.setContentsMargins(8, 8, 8, 8)

        # Left: image view
        self.img_view = pg.GraphicsLayoutWidget()
        self.img_plot = self.img_view.addPlot(row=0, col=0)
        self.img_plot.setAspectLocked(True)
        self.img_plot.setLabel('bottom', 'x (px)')
        self.img_plot.setLabel('left', 'y (px)')
        self.img_plot.invertY(True)
        self.img_item = pg.ImageItem()
        self.img_plot.addItem(self.img_item)

        # ROI overlay (square, resizable)
        self.roi = pg.RectROI([50, 50], [self.roi_size, self.roi_size],
                              pen=pg.mkPen(0, 255, 0, width=1.2),
                              movable=True, resizable=True)
        self.roi.addScaleHandle([1, 1], [0, 0])
        self.roi.addScaleHandle([0, 0], [1, 1])
        self.roi.setZValue(10)
        self.img_plot.addItem(self.roi)
        self.roi.sigRegionChanged.connect(self.on_roi_changed)
        self.roi.sigRegionChangeFinished.connect(self.on_roi_changed)

        # Zoomed ROI view
        self.zoom_view = pg.GraphicsLayoutWidget()
        self.zoom_plot = self.zoom_view.addPlot(row=0, col=0)
        self.zoom_plot.setAspectLocked(True)
        self.zoom_plot.setTitle("ROI Zoom")
        self.zoom_plot.setLabel('bottom', 'x (px)')
        self.zoom_plot.setLabel('left', 'y (px)')
        self.zoom_plot.invertY(True)
        self.zoom_item = pg.ImageItem()
        self.zoom_plot.addItem(self.zoom_item)
        self._line_profile_box_item = pg.PlotDataItem(
            pen=pg.mkPen(255, 160, 40, width=1.2)
        )
        self.zoom_plot.addItem(self._line_profile_box_item)
        self._line_profile_box_item.setVisible(False)
        self._line_profile_click_marker = pg.PlotDataItem(
            pen=pg.mkPen(255, 220, 120, width=2)
        )
        self.zoom_plot.addItem(self._line_profile_click_marker)
        self._line_profile_click_marker.setVisible(False)
        self.zoom_view.setMinimumSize(400, 400)

        # Middle column: ROI zoom + FFT
        middle_col = QtWidgets.QWidget()
        middle_layout = QtWidgets.QVBoxLayout(middle_col)
        middle_layout.setContentsMargins(0, 0, 0, 0)
        middle_layout.setSpacing(6)

        # FFT view
        self.fft_view = pg.GraphicsLayoutWidget()
        self.fft_view.setMinimumSize(400, 400)  # UI square-ish
        self.fft_plot = self.fft_view.addPlot(row=0, col=0)
        self.fft_plot.setAspectLocked(True)  # no stretching
        self.fft_plot.setTitle("FFT (ROI)")
        self.fft_plot.setLabel('bottom', 'k_x (nm⁻¹)')
        self.fft_plot.setLabel('left', 'k_y (nm⁻¹)')
        self.fft_plot.invertY(True)
        self.fft_item = pg.ImageItem()
        self.fft_plot.addItem(self.fft_item)
        self.fft_item.setRect(QtCore.QRectF(-1.0, -1.0, 2.0, 2.0))

        # Center crosshair at k=0
        self.vline = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen(255, 255, 0, width=1))
        self.hline = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen(255, 255, 0, width=1))
        self.fft_plot.addItem(self.vline)
        self.fft_plot.addItem(self.hline)
        self.vline.setVisible(False)
        self.hline.setVisible(False)

        # Scatter points
        self.scatter = pg.ScatterPlotItem(
            size=10,
            pen=pg.mkPen(255, 0, 0, width=2),
            brush=pg.mkBrush(255, 0, 0, 120)
        )
        self.fft_plot.addItem(self.scatter)

        # Ring marker (set from radial-profile click)
        self.ring_item = QtWidgets.QGraphicsEllipseItem()
        self.ring_item.setPen(pg.mkPen(80, 220, 255, width=1.5))
        self.ring_item.setBrush(pg.mkBrush(0, 0, 0, 0))
        self.ring_item.setVisible(False)
        self.fft_plot.addItem(self.ring_item)

        # Click handling
        self.fft_plot.scene().sigMouseClicked.connect(self.on_fft_clicked)

        # Radial profile view
        self.radial_plot = pg.PlotWidget()
        self.radial_plot.setMinimumHeight(170)
        self.radial_plot.setTitle("Radial Profile")
        self.radial_plot.setLabel('bottom', '|k| (nm⁻¹)')
        self.radial_plot.setLabel('left', 'Mean FFT Intensity')
        self.radial_curve = self.radial_plot.plot([], [], pen=pg.mkPen(80, 220, 255, width=2))
        self._radial_vline_item = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen(255, 140, 0, width=1.2))
        self._radial_vline_item.setVisible(False)
        self.radial_plot.addItem(self._radial_vline_item)
        self._profile_readout_item = pg.TextItem(text="", anchor=(0.5, 0.5), color=(255, 220, 120))
        self._profile_readout_item.setVisible(False)
        self.radial_plot.addItem(self._profile_readout_item, ignoreBounds=True)
        self._profile_range = None
        self._profile_click_x = None  # backward compatibility for old metadata
        self.radial_plot.scene().sigMouseClicked.connect(self.on_radial_clicked)
        self.radial_plot.getPlotItem().vb.sigRangeChanged.connect(lambda *_: self._update_profile_readout_position())
        self._profile_region_item = pg.LinearRegionItem(
            values=[0.0, 1.0],
            orientation=pg.LinearRegionItem.Vertical,
            movable=True,
            brush=pg.mkBrush(255, 180, 80, 40),
            pen=pg.mkPen(255, 180, 80, width=1.2),
        )
        self._profile_region_item.setVisible(False)
        self._profile_region_item.sigRegionChanged.connect(self.on_profile_region_changed)
        self.radial_plot.addItem(self._profile_region_item)

        # Controls
        ctrl = QtWidgets.QGroupBox("Controls / Calibration")
        ctrl_layout = QtWidgets.QGridLayout(ctrl)

        self.edit_load_path = QtWidgets.QLineEdit()
        self.edit_load_path.setPlaceholderText("Full file path, e.g. /path/to/image.dm4")
        self.btn_browse = QtWidgets.QPushButton("Browse...")
        self.btn_browse.clicked.connect(self.browse_file)
        self.edit_load_path.returnPressed.connect(self.open_image)
        self.btn_open = QtWidgets.QPushButton("Open Image")
        self.btn_open.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.btn_open.clicked.connect(self.open_image)

        self.btn_open_svg = QtWidgets.QPushButton("Open SVG")
        self.btn_open_svg.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.btn_open_svg.clicked.connect(self.open_svg)

        self.btn_save_svg_300 = QtWidgets.QPushButton("Save SVG (300 dpi)")
        self.btn_save_svg_600 = QtWidgets.QPushButton("Save SVG (600 dpi)")
        self.btn_save_svg_300.clicked.connect(lambda: self.save_combined_svg(dpi=300))
        self.btn_save_svg_600.clicked.connect(lambda: self.save_combined_svg(dpi=600))

        self.btn_scale = QtWidgets.QPushButton("Calibrate Scale")
        self.btn_scale.clicked.connect(self.on_scale_button_clicked)
        self.edit_scale_display = QtWidgets.QLineEdit("-- nm/px")
        self.edit_scale_display.setReadOnly(True)
        self.edit_scale_display.returnPressed.connect(self.on_scale_display_return_pressed)

        self.btn_clear_points = QtWidgets.QPushButton("Clear Picks")
        self.btn_clear_points.clicked.connect(self.clear_picks)

        self.btn_remove_last = QtWidgets.QPushButton("Remove Last Pick")
        self.btn_remove_last.clicked.connect(self.remove_last_pick)

        self.spin_roi = QtWidgets.QSpinBox()
        self.spin_roi.setRange(64, 4096)
        self.spin_roi.setSingleStep(64)
        self.spin_roi.setValue(self.roi_size)
        self.spin_roi.valueChanged.connect(self.set_roi_size)

        self.spin_roi_angle = QtWidgets.QDoubleSpinBox()
        self.spin_roi_angle.setRange(-180.0, 180.0)
        self.spin_roi_angle.setSingleStep(0.5)
        self.spin_roi_angle.setDecimals(1)
        self.spin_roi_angle.setValue(0.0)
        self.spin_roi_angle.setSuffix(" deg")
        self.spin_roi_angle.valueChanged.connect(self.set_roi_angle)

        # Checkbox for bin2 FFT
        self.chk_bin2 = QtWidgets.QCheckBox("Bin2 before FFT")
        self.chk_bin2.setChecked(False)
        self.chk_bin2.stateChanged.connect(self.on_bin2_changed)

        # Checkbox for bin4 FFT
        self.chk_bin4 = QtWidgets.QCheckBox("Bin4 before FFT")
        self.chk_bin4.setChecked(True)
        self.chk_bin4.stateChanged.connect(self.on_bin4_changed)

        # Auto refresh contrast/brightness for FFT display
        self.chk_autocontrast = QtWidgets.QCheckBox("Auto contrast (FFT)")
        self.chk_autocontrast.setChecked(True)
        self.chk_autocontrast.stateChanged.connect(lambda _=None: self.update_fft())

        # Log-y display for radial profile
        self.chk_logy = QtWidgets.QCheckBox("Radial profile: log y")
        self.chk_logy.setChecked(False)
        self.chk_logy.stateChanged.connect(lambda _=None: self.update_radial_plot())

        # Toggle between FFT radial profile and ROI line profile
        self.chk_image_profile = QtWidgets.QCheckBox("Profile from ROI line")
        self.chk_image_profile.setChecked(False)
        self.chk_image_profile.stateChanged.connect(self.on_profile_mode_changed)

        self.spin_profile_width = QtWidgets.QSpinBox()
        self.spin_profile_width.setRange(1, 200)
        self.spin_profile_width.setSingleStep(1)
        self.spin_profile_width.setValue(5)
        self.spin_profile_width.valueChanged.connect(self.on_line_profile_width_changed)

        # store last applied FFT display levels
        self._fft_levels = None  # (lo, hi)

        # Pixel size display (fixed, read-only)
        self.edit_px = QtWidgets.QDoubleSpinBox()
        self.edit_px.setRange(0.0, 100000.0)
        self.edit_px.setDecimals(6)
        self.edit_px.setValue(0.0)
        self.edit_px.setSuffix(" nm/px")
        self.edit_px.setReadOnly(True)
        self.edit_px.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        self.edit_px.setFocusPolicy(QtCore.Qt.NoFocus)
        self.edit_px.setVisible(False)

        row = 0
        ctrl_layout.addWidget(self.btn_browse, row, 0)
        ctrl_layout.addWidget(self.edit_load_path, row, 1, 1, 2); row += 1
        ctrl_layout.setColumnStretch(0, 0)
        ctrl_layout.setColumnStretch(1, 1)
        ctrl_layout.setColumnStretch(2, 0)
        ctrl_layout.addWidget(self.btn_open, row, 0)
        ctrl_layout.addWidget(self.btn_open_svg, row, 1); row += 1
        ctrl_layout.addWidget(self.btn_save_svg_300, row, 0)
        ctrl_layout.addWidget(self.btn_save_svg_600, row, 1); row += 1
        ctrl_layout.addWidget(self.btn_scale, row, 0)
        ctrl_layout.addWidget(self.edit_scale_display, row, 1); row += 1
        ctrl_layout.addWidget(QtWidgets.QLabel("ROI size (px):"), row, 0)
        ctrl_layout.addWidget(self.spin_roi, row, 1); row += 1
        ctrl_layout.addWidget(QtWidgets.QLabel("ROI angle:"), row, 0)
        ctrl_layout.addWidget(self.spin_roi_angle, row, 1); row += 1
        ctrl_layout.addWidget(self.chk_bin2, row, 0, 1, 2); row += 1
        ctrl_layout.addWidget(self.chk_bin4, row, 0, 1, 2); row += 1
        ctrl_layout.addWidget(self.chk_autocontrast, row, 0, 1, 2); row += 1
        ctrl_layout.addWidget(self.chk_logy, row, 0, 1, 2); row += 1
        ctrl_layout.addWidget(self.chk_image_profile, row, 0, 1, 2); row += 1
        ctrl_layout.addWidget(QtWidgets.QLabel("Line width (px):"), row, 0)
        ctrl_layout.addWidget(self.spin_profile_width, row, 1); row += 1
        ctrl_layout.addWidget(self.btn_clear_points, row, 0)
        ctrl_layout.addWidget(self.btn_remove_last, row, 1); row += 1

        # Table: |k|, d, and angle
        self.table = QtWidgets.QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["#", "|k| (nm⁻¹)", "d (nm)", "∠(deg)"])
        # self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.configure_table_columns()

        side_col = QtWidgets.QWidget()
        side_layout = QtWidgets.QVBoxLayout(side_col)
        side_layout.setContentsMargins(0, 0, 0, 0)
        side_layout.setSpacing(10)

        middle_layout.addWidget(self.zoom_view, stretch=1)
        middle_layout.addWidget(self.fft_view, stretch=1)
        side_layout.addWidget(ctrl, stretch=0)
        side_layout.addWidget(self.table, stretch=1)

        # Splitter
        left_col = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_col)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(4)
        left_layout.addWidget(self.img_view, stretch=7)
        left_layout.addWidget(self.radial_plot, stretch=3)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        splitter.addWidget(left_col)
        splitter.addWidget(middle_col)
        splitter.addWidget(side_col)
        # Match the preferred layout ratio: left > middle > right.
        splitter.setSizes([760, 560, 360])
        main_layout.addWidget(splitter)

        # Visual defaults
        # pg.setConfigOptions(imageAxisOrder='row-major')

        # Initial FFT display
        self.update_fft()

        # Undo shortcut: Ctrl+Z / Command+Z
        self._sc_undo = QtWidgets.QShortcut(QtGui.QKeySequence.Undo, self)
        self._sc_undo.activated.connect(self.remove_last_pick)

        # Close window shortcut: Ctrl+W / Command+W
        self._sc_close = QtWidgets.QShortcut(QtGui.QKeySequence.Close, self)
        self._sc_close.activated.connect(self.close)

    # -------------------------
    # Open / Display
    # -------------------------
    def _current_input_path(self) -> Path:
        path_text = self._normalized_input_text()
        if not path_text:
            raise ValueError("Please input a full file path.")
        path = Path(path_text).expanduser()
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"File not found: {path}")
        return path

    def _normalized_input_text(self) -> str:
        """Accept copied paths wrapped in single or double quotes."""
        path_text = self.edit_load_path.text().strip()
        if len(path_text) >= 2 and path_text[0] == path_text[-1] and path_text[0] in {"'", '"'}:
            path_text = path_text[1:-1].strip()
        return path_text

    def _choose_image_file_from_input(self) -> Path | None:
        """
        Return a file path from the current input.
        - file path: use it directly
        - folder path: open a file dialog from that folder
        """
        path_text = self._normalized_input_text()
        if not path_text:
            start_dir = "."
        else:
            candidate = Path(path_text).expanduser()
            if candidate.exists() and candidate.is_file():
                return candidate
            start_dir = str(candidate if candidate.exists() and candidate.is_dir() else candidate.parent)

        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select image",
            start_dir,
            "Supported (*.emd *.dm4 *.dm3 *.tif *.tiff *.emi *.jpg *.jpeg *.png *.mrc);;All files (*.*)",
        )
        if not path:
            return None
        selected = Path(path).expanduser()
        self.edit_load_path.setText(str(selected))
        return selected

    @staticmethod
    def _is_plain_image_suffix(path: Path) -> bool:
        return path.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".mrc"}

    @staticmethod
    def _load_plain_image(path: Path) -> np.ndarray:
        ext = path.suffix.lower()
        if ext in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}:
            from PIL import Image
            arr = np.asarray(Image.open(str(path)))
            if arr.ndim == 3:
                arr = arr[..., :3]
                arr = 0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]
            elif arr.ndim > 3:
                arr = arr.reshape((-1,) + arr.shape[-2:])[0]
            return np.asarray(arr, dtype=np.float32)
        if ext == ".mrc":
            import mrcfile
            with mrcfile.open(str(path), permissive=True) as m:
                arr = np.asarray(m.data)
            if arr.ndim > 2:
                arr = arr.reshape((-1,) + arr.shape[-2:])[0]
            return np.asarray(arr, dtype=np.float32)
        raise ValueError(f"Unsupported plain image format: {path.suffix}")

    def _image_preview_title(self) -> str:
        if not self.img_path:
            return ""
        return Path(self.img_path).expanduser().name

    def browse_file(self):
        start_dir = ""
        try:
            current = Path(self._normalized_input_text() or ".").expanduser()
            start_dir = str(current.parent if current.suffix else current)
        except Exception:
            start_dir = "."

        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select image",
            start_dir,
            "Supported (*.emd *.dm4 *.dm3 *.tif *.tiff *.emi *.jpg *.jpeg *.png *.mrc);;All files (*.*)",
        )
        if not path:
            return
        self.edit_load_path.setText(str(Path(path).expanduser()))

    def open_image(self):
        try:
            data_path = self._choose_image_file_from_input()
            if data_path is None:
                return
            key = data_path.name
            if self._is_plain_image_suffix(data_path):
                arr = self._load_plain_image(data_path)
                sub = {"data": arr, "axes": []}
            else:
                sub = load_hyperspy_dataset(data_path)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Load failed", str(e))
            return

        self._apply_loaded_subdataset(
            sub=sub,
            data_path=Path(data_path),
            key=key,
        )

    def _apply_loaded_subdataset(self, sub: dict, data_path: Path, key: str):
        arr = np.asarray(sub.get("data", None))
        if arr is None or arr.size == 0:
            QtWidgets.QMessageBox.critical(self, "Load failed", "Dataset has empty data.")
            return

        # If stack, take first frame
        if arr.ndim > 2:
            arr = arr.reshape((-1,) + arr.shape[-2:])[0]

        self.img = to_float_image(arr)
        self.img_path = str(data_path)
        self.edit_load_path.setText(str(data_path))

        self.img_item.setImage(self.img, autoLevels=True)
        self.img_plot.setTitle(self._image_preview_title())

        # Read pixel size from sub['axes'] (scale/units)
        nm_per_px = 0.0
        try:
            axes = sub.get("axes", [])
            # Try to pick a spatial axis dict that has 'scale' and 'units'
            ax = {}
            if isinstance(axes, (list, tuple)):
                # prefer the last two axes if present (common for 2D)
                candidates = list(axes)[-2:] if len(axes) >= 2 else list(axes)
                for cand in candidates:
                    if isinstance(cand, dict) and ("scale" in cand) and ("units" in cand):
                        ax = cand
                        break
                if not ax and len(axes) > 0 and isinstance(axes[0], dict):
                    ax = axes[0]
            elif isinstance(axes, dict):
                ax = axes

            nm_per_px = _to_nm_per_px(ax.get("scale", None), ax.get("units", None))
        except Exception:
            nm_per_px = 0.0

        self._set_pixel_scale_nm(float(nm_per_px), source=("metadata" if nm_per_px > 0 else "none"))
        self._exit_scale_calibration_mode(keep_text=True)

        # Reset ROI to center (square, clamped, even size)
        h, w = self.img.shape[:2]
        s = int(min(self.roi_size, h, w))
        s = (s // 2) * 2  # even; bin4 path trims to /4 internally
        x0 = (w - s) // 2
        y0 = (h - s) // 2
        x0, y0, s = _clamp_roi_square(x0, y0, s, w, h)

        self.roi.blockSignals(True)
        self.roi.setPos([x0, y0])
        self.roi.setSize([s, s])
        self._apply_roi_angle()
        self.roi.blockSignals(False)

        self.clear_picks()
        self.update_fft()

    def _set_pixel_scale_nm(self, nm_per_px: float, source: str = "none"):
        try:
            sc = float(nm_per_px)
        except Exception:
            sc = 0.0
        if (not np.isfinite(sc)) or sc <= 0:
            sc = 0.0
        self.edit_px.blockSignals(True)
        self.edit_px.setValue(sc)
        self.edit_px.blockSignals(False)

        if sc > 0 and source == "metadata":
            self._scale_source = "metadata"
            if not self._calibrating_scale:
                self.btn_scale.setText("Scale (Metadata)")
        elif sc > 0 and source == "calibrated":
            self._scale_source = "calibrated"
            if not self._calibrating_scale:
                self.btn_scale.setText("Calibrated Scale")
        else:
            self._scale_source = "none"
            if not self._calibrating_scale:
                self.btn_scale.setText("Calibrate Scale")

        if not self._calibrating_scale:
            self.edit_scale_display.setReadOnly(True)
            self.edit_scale_display.setText(f"{sc:.6g} nm/px" if sc > 0 else "-- nm/px")
        self._apply_image_axis_scale()

    def _apply_image_axis_scale(self):
        dx_nm = float(self.edit_px.value())
        if dx_nm > 0:
            axb = self.img_plot.getAxis('bottom')
            ayl = self.img_plot.getAxis('left')
            axb.setLabel('x (nm)')
            ayl.setLabel('y (nm)')
            axb.setScale(dx_nm)
            ayl.setScale(dx_nm)
        else:
            self.img_plot.setLabel('bottom', 'x (px)')
            self.img_plot.setLabel('left', 'y (px)')
            self.img_plot.getAxis('bottom').setScale(1.0)
            self.img_plot.getAxis('left').setScale(1.0)

    def _calibration_line_length_px(self):
        if self._calib_line_roi is None:
            return 0.0
        try:
            return abs(float(self._calib_line_roi.size().x()))
        except Exception:
            return 0.0

    def _set_line_segment_roi_points(self, roi, p1, p2):
        if roi is None:
            return
        x1, y1 = float(p1[0]), float(p1[1])
        x2, y2 = float(p2[0]), float(p2[1])
        if hasattr(roi, "setPoints"):
            roi.setPoints([[x1, y1], [x2, y2]])
            return
        handles = roi.listPoints()
        if handles is None or len(handles) < 2:
            raise RuntimeError("LineSegmentROI has insufficient handles.")
        roi.movePoint(0, pg.Point(x1, y1), finish=False)
        roi.movePoint(1, pg.Point(x2, y2), finish=True)

    def _on_calibration_line_changed(self):
        if self._calib_line_roi is None or self._updating_calib_line:
            return
        try:
            pos = self._calib_line_roi.pos()
            size = self._calib_line_roi.size()
            x = float(pos.x())
            y = float(pos.y())
            w = float(size.x())
            h = float(size.y())
        except Exception:
            return
        target_h = float(max(1.0, self._calib_bar_h_px))
        target_w = float(max(1.0, abs(w)))
        if (abs(h - target_h) <= 1e-6) and (abs(w - target_w) <= 1e-6):
            return
        self._updating_calib_line = True
        try:
            self._calib_line_roi.setPos([x, y])
            self._calib_line_roi.setSize([target_w, target_h])
        finally:
            self._updating_calib_line = False

    def _ensure_horizontal_calib_line_hooks(self):
        if self._calib_line_roi is None:
            return
        try:
            self._calib_line_roi.sigRegionChanged.disconnect(self._on_calibration_line_changed)
        except Exception:
            pass
        try:
            self._calib_line_roi.sigRegionChangeFinished.disconnect(self._on_calibration_line_changed)
        except Exception:
            pass
        self._calib_line_roi.sigRegionChanged.connect(self._on_calibration_line_changed)
        self._calib_line_roi.sigRegionChangeFinished.connect(self._on_calibration_line_changed)

    def _enter_scale_calibration_mode(self):
        if self.img is None:
            return
        self._calibrating_scale = True
        self.btn_scale.setText("Apply Scale")
        self.edit_scale_display.setReadOnly(False)
        self.edit_scale_display.clear()
        self.edit_scale_display.setPlaceholderText("nm")
        self.edit_scale_display.setFocus()
        if self._calib_line_roi is None:
            h, w = self.img.shape[:2]
            bar_h = float(np.clip(0.015 * h, 3.0, 8.0))
            y_center = float(np.clip(0.88 * h, 0.0, max(0.0, h - 1.0)))
            y = float(np.clip(y_center - 0.5 * bar_h, 0.0, max(0.0, h - bar_h)))
            line_len = float(np.clip(0.25 * w, 10.0, min(200.0, max(10.0, w))))
            x1 = 0.5 * (w - line_len)
            self._calib_bar_h_px = bar_h
            self._calib_line_roi = pg.RectROI(
                [x1, y],
                [line_len, bar_h],
                pen=pg.mkPen(255, 210, 80, width=2),
                movable=True,
                rotatable=False,
            )
            self._calib_line_roi.addScaleHandle([1.0, 0.5], [0.0, 0.5])
            self._calib_line_roi.addScaleHandle([0.0, 0.5], [1.0, 0.5])
            self.img_plot.addItem(self._calib_line_roi)
        else:
            try:
                self._calib_bar_h_px = float(max(1.0, self._calib_line_roi.size().y()))
            except Exception:
                pass
        self._ensure_horizontal_calib_line_hooks()
        self._calib_line_roi.setVisible(True)
        self._on_calibration_line_changed()

    def _exit_scale_calibration_mode(self, keep_text: bool = False):
        self._calibrating_scale = False
        if self._calib_line_roi is not None:
            self._calib_line_roi.setVisible(False)
        self.edit_scale_display.setPlaceholderText("")
        if not keep_text:
            sc = float(self.edit_px.value())
            self.edit_scale_display.setText(f"{sc:.6g} nm/px" if sc > 0 else "-- nm/px")
        self.edit_scale_display.setReadOnly(True)
        if self._scale_source == "metadata" and float(self.edit_px.value()) > 0:
            self.btn_scale.setText("Scale (Metadata)")
        elif self._scale_source == "calibrated" and float(self.edit_px.value()) > 0:
            self.btn_scale.setText("Calibrated Scale")
        else:
            self.btn_scale.setText("Calibrate Scale")

    def on_scale_display_return_pressed(self):
        if self._calibrating_scale:
            self._apply_scale_calibration()

    def on_scale_button_clicked(self):
        if not self._calibrating_scale:
            self._enter_scale_calibration_mode()
            return
        self._apply_scale_calibration()

    def _apply_scale_calibration(self):
        txt = self.edit_scale_display.text().strip()
        try:
            m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", txt)
            if m is None:
                raise ValueError("no numeric value")
            real_len = float(m.group(0))
        except Exception:
            QtWidgets.QMessageBox.warning(self, "Calibration", "Please input a numeric real length (nm).")
            return
        if not np.isfinite(real_len) or real_len <= 0:
            QtWidgets.QMessageBox.warning(self, "Calibration", "Real length must be > 0.")
            return
        px_len = self._calibration_line_length_px()
        if px_len <= 1e-9:
            QtWidgets.QMessageBox.warning(self, "Calibration", "Calibration line length is too small.")
            return
        nm_per_px = float(real_len / px_len)
        self._set_pixel_scale_nm(nm_per_px, source="calibrated")
        self._exit_scale_calibration_mode(keep_text=False)
        self.update_fft()

    @staticmethod
    def _read_svg_metadata(svg_path: Path):
        with open(svg_path, "r", encoding="utf-8") as f:
            svg = f.read()
        s = svg.find("<metadata>")
        e = svg.find("</metadata>")
        if s < 0 or e < 0 or e <= s:
            raise ValueError("No <metadata>...</metadata> found in SVG.")
        meta_json = svg[s + len("<metadata>"):e].strip()
        return json.loads(meta_json)

    @staticmethod
    def _inject_svg_metadata(svg_path: Path, meta: dict):
        import re

        meta_json = json.dumps(meta, ensure_ascii=False)
        with open(svg_path, "r", encoding="utf-8") as f:
            svg = f.read()
        if "<metadata>" in svg:
            svg = re.sub(
                r"<metadata>.*?</metadata>",
                f"<metadata>{meta_json}</metadata>",
                svg,
                flags=re.DOTALL,
            )
        else:
            idx = svg.find(">") + 1
            svg = svg[:idx] + f"\n<metadata>{meta_json}</metadata>\n" + svg[idx:]
        with open(svg_path, "w", encoding="utf-8") as f:
            f.write(svg)

    def _metadata_dict(self):
        pos = self.roi.pos()
        size = self.roi.size()
        bin_factor = 1
        if self.chk_bin4.isChecked():
            bin_factor = 4
        elif self.chk_bin2.isChecked():
            bin_factor = 2
        line_seg = self._line_profile_endpoints()
        line_seg_dict = None
        if line_seg is not None:
            x1, y1, x2, y2 = line_seg
            line_seg_dict = {"p1": [float(x1), float(y1)], "p2": [float(x2), float(y2)]}
        return {
            "tool": "TEM_Tools.hrtem_analyser",
            "metadata_version": 1,
            "root_path": str(Path(self.img_path).expanduser().parent) if self.img_path else "",
            "key": Path(self.img_path).name if self.img_path else "",
            "data_path": self.img_path,
            "pixel_size_nm_per_px": float(self.edit_px.value()),
            "roi": {
                "x": float(pos.x()),
                "y": float(pos.y()),
                "size": float(min(size.x(), size.y())),
                "angle_deg": float(self.roi_angle_deg),
            },
            "fft": {
                "bin_factor": int(bin_factor),
                "autocontrast": bool(self.chk_autocontrast.isChecked()),
                "fft_levels": (
                    [float(self._fft_levels[0]), float(self._fft_levels[1])]
                    if isinstance(self._fft_levels, (list, tuple)) and len(self._fft_levels) >= 2
                    else None
                ),
            },
            "profile": {
                "mode": "line" if self._profile_mode_image else "radial",
                "log_y": bool(self.chk_logy.isChecked()),
                "line_width_px": int(self.spin_profile_width.value()),
                "ring_radius_nm_inv": (
                    float(self._ring_radius) if self._ring_radius is not None else None
                ),
                "line_click_distance": (
                    float(self._profile_click_x) if self._profile_click_x is not None else None
                ),
                "line_distance_range": (
                    [float(self._profile_range[0]), float(self._profile_range[1])]
                    if isinstance(self._profile_range, (list, tuple)) and len(self._profile_range) == 2
                    else None
                ),
                "line_segment": line_seg_dict,
            },
            "picks": [
                {"kx": float(p["kx"]), "ky": float(p["ky"])}
                for p in self.points
            ],
        }

    def open_svg(self):
        start_dir = ""
        try:
            current = self._current_input_path()
            start_dir = str(current.parent)
        except Exception:
            start_dir = str(Path(self._normalized_input_text() or ".").expanduser())

        svg_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open SVG",
            start_dir,
            "SVG (*.svg)",
        )
        if not svg_path:
            return
        try:
            meta = self._read_svg_metadata(Path(svg_path))
            self._load_from_svg_metadata(meta)
            QtWidgets.QMessageBox.information(
                self, "Restored", f"Restored experiment from:\n{Path(svg_path).name}"
            )
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Open SVG failed", str(e))

    def _load_from_svg_metadata(self, meta: dict):
        data_path = meta.get("data_path", None)
        sub = None
        used_data_path = None
        if data_path:
            p = Path(str(data_path)).expanduser()
            if p.exists() and p.is_file():
                if self._is_plain_image_suffix(p):
                    arr = self._load_plain_image(p)
                    sub = {"data": arr, "axes": []}
                    used_data_path = p
                else:
                    sub = load_hyperspy_dataset(p)
                    used_data_path = p

        if sub is None:
            raise ValueError("SVG metadata does not contain a valid source file path.")

        self._apply_loaded_subdataset(
            sub=sub,
            data_path=used_data_path,
            key=used_data_path.name,
        )
        self._restore_gui_from_metadata(meta)

    def _restore_gui_from_metadata(self, meta: dict):
        def _safe_float(v, default=None):
            try:
                return float(v)
            except Exception:
                return default

        fft_meta = meta.get("fft", {}) or {}
        profile_meta = meta.get("profile", {}) or {}
        roi_meta = meta.get("roi", {}) or {}

        with QtCore.QSignalBlocker(self.chk_bin2), QtCore.QSignalBlocker(self.chk_bin4):
            bf = int(fft_meta.get("bin_factor", 4))
            self.chk_bin2.setChecked(bf == 2)
            self.chk_bin4.setChecked(bf == 4)
        with QtCore.QSignalBlocker(self.chk_autocontrast):
            self.chk_autocontrast.setChecked(bool(fft_meta.get("autocontrast", True)))
        with QtCore.QSignalBlocker(self.chk_logy):
            self.chk_logy.setChecked(bool(profile_meta.get("log_y", False)))
        with QtCore.QSignalBlocker(self.spin_profile_width):
            w = int(np.clip(int(profile_meta.get("line_width_px", self.spin_profile_width.value())), 1, 200))
            self.spin_profile_width.setValue(w)
        with QtCore.QSignalBlocker(self.chk_image_profile):
            self.chk_image_profile.setChecked(str(profile_meta.get("mode", "radial")).lower() == "line")
        self._profile_mode_image = bool(self.chk_image_profile.isChecked())

        ang = _safe_float(roi_meta.get("angle_deg", None), 0.0)
        with QtCore.QSignalBlocker(self.spin_roi_angle):
            self.spin_roi_angle.setValue(float(np.clip(ang, -180.0, 180.0)))
        self.roi_angle_deg = float(self.spin_roi_angle.value())

        if self.img is not None:
            h, w = self.img.shape[:2]
            s = _safe_float(roi_meta.get("size", None), float(self.spin_roi.value()))
            x0 = _safe_float(roi_meta.get("x", None), float(self.roi.pos().x()))
            y0 = _safe_float(roi_meta.get("y", None), float(self.roi.pos().y()))
            x0, y0, s = _clamp_roi_square(x0, y0, s, w, h)
            with QtCore.QSignalBlocker(self.spin_roi):
                self.spin_roi.setValue(int(max(64, min(4096, round(s)))))
            self.roi.blockSignals(True)
            self.roi.setPos([x0, y0])
            self.roi.setSize([s, s])
            self._apply_roi_angle()
            self.roi.blockSignals(False)

        levels = fft_meta.get("fft_levels", None)
        if isinstance(levels, (list, tuple)) and len(levels) >= 2:
            lo = _safe_float(levels[0], None)
            hi = _safe_float(levels[1], None)
            if lo is not None and hi is not None and hi > lo:
                self._fft_levels = (lo, hi)

        self.update_fft()

        line_seg = profile_meta.get("line_segment", None)
        if isinstance(line_seg, dict) and self._line_profile_roi is not None:
            p1 = line_seg.get("p1", None)
            p2 = line_seg.get("p2", None)
            if isinstance(p1, (list, tuple)) and isinstance(p2, (list, tuple)) and len(p1) >= 2 and len(p2) >= 2:
                try:
                    self._set_line_segment_roi_points(
                        self._line_profile_roi,
                        (float(p1[0]), float(p1[1])),
                        (float(p2[0]), float(p2[1])),
                    )
                except Exception:
                    pass

        ring = _safe_float(profile_meta.get("ring_radius_nm_inv", None), None)
        self._ring_radius = ring if (ring is not None and ring > 0) else None
        range_vals = profile_meta.get("line_distance_range", None)
        if isinstance(range_vals, (list, tuple)) and len(range_vals) >= 2:
            rs = _safe_float(range_vals[0], None)
            re = _safe_float(range_vals[1], None)
            if rs is not None and re is not None:
                self._profile_range = (min(rs, re), max(rs, re))
            else:
                self._profile_range = None
        else:
            click_x = _safe_float(profile_meta.get("line_click_distance", None), None)
            self._profile_click_x = click_x if click_x is not None else None
            self._profile_range = None

        self.clear_picks()
        for p in (meta.get("picks", []) or []):
            kx = _safe_float((p or {}).get("kx", None), None)
            ky = _safe_float((p or {}).get("ky", None), None)
            if kx is None or ky is None:
                continue
            k = float(np.hypot(kx, ky))
            if k <= 1e-12:
                continue
            self.points.append({"kx": float(kx), "ky": float(ky), "k": k, "d": 1.0 / k})

        self.update_scatter()
        for t in self.text_labels:
            try:
                self.fft_plot.removeItem(t)
            except Exception:
                pass
        self.text_labels = []
        for i, p in enumerate(self.points, start=1):
            t = pg.TextItem(text=str(i), anchor=(0, 1), color=(255, 80, 80))
            t.setPos(float(p["kx"]), float(p["ky"]))
            self.fft_plot.addItem(t)
            self.text_labels.append(t)
        self.refresh_table_all()
        self.update_line_profile_box_overlay()
        if self._profile_mode_image:
            self.update_line_profile_from_roi()
        self.update_ring_on_fft()
        self.update_radial_plot()

    def save_combined_svg(self, dpi: int = 600):
        try:
            import matplotlib.pyplot as plt
            from matplotlib.patches import Rectangle
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Save failed", f"matplotlib import failed:\n{e}")
            return
        dpi = int(dpi)
        if dpi <= 0:
            dpi = 600

        key_text = Path(self.img_path).name if self.img_path else ""
        key_stem = Path(key_text).stem if key_text else ""
        if not key_stem:
            key_stem = "unknown"
        default_name = f"Image_{key_stem}.svg"

        default_dir = Path(self.img_path).expanduser().parent if self.img_path else Path(".").expanduser()
        default_path = str(default_dir / default_name)

        out_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save Combined SVG",
            default_path,
            "SVG files (*.svg)"
        )
        if not out_path:
            return
        out_path_obj = Path(out_path).expanduser()
        if out_path_obj.suffix.lower() != ".svg":
            out_path_obj = out_path_obj.with_suffix(".svg")
        # Keep export file name fixed as Image_{key}.svg, only use selected folder.
        out_path_obj = out_path_obj.parent / default_name

        if out_path_obj.exists():
            msg = QtWidgets.QMessageBox(self)
            msg.setIcon(QtWidgets.QMessageBox.Question)
            msg.setWindowTitle("File exists")
            msg.setText(f"File already exists:\n{out_path_obj.name}")
            msg.setInformativeText("Replace it, or save as a renamed file with suffix _number?")
            btn_replace = msg.addButton("Replace", QtWidgets.QMessageBox.AcceptRole)
            btn_rename = msg.addButton("Rename (_number)", QtWidgets.QMessageBox.ActionRole)
            btn_cancel = msg.addButton("Cancel", QtWidgets.QMessageBox.RejectRole)
            msg.setDefaultButton(btn_rename)
            msg.exec_()
            clicked = msg.clickedButton()
            if clicked is btn_cancel:
                return
            if clicked is btn_rename:
                base = out_path_obj.stem
                suffix = out_path_obj.suffix
                idx = 1
                while True:
                    candidate = out_path_obj.parent / f"{base}_{idx}{suffix}"
                    if not candidate.exists():
                        out_path_obj = candidate
                        break
                    idx += 1

        out_path = str(out_path_obj)

        px_nm = float(self.edit_px.value())

        # Keep export brightness/contrast aligned with current GUI display levels.
        img_levels = None
        fft_levels = None
        try:
            if hasattr(self.img_item, "getLevels"):
                img_levels = self.img_item.getLevels()
            elif hasattr(self.img_item, "levels"):
                img_levels = self.img_item.levels
        except Exception:
            img_levels = None
        try:
            if hasattr(self.fft_item, "getLevels"):
                fft_levels = self.fft_item.getLevels()
            elif hasattr(self.fft_item, "levels"):
                fft_levels = self.fft_item.levels
        except Exception:
            fft_levels = None

        fft_mag = None
        kmax = float(self._fft_kmax) if self._fft_kmax is not None else 0.5
        try:
            if getattr(self.fft_item, "image", None) is not None:
                fft_mag = np.asarray(self.fft_item.image, dtype=np.float32)
        except Exception:
            fft_mag = None
        if fft_mag is None or fft_mag.size == 0:
            fft_mag = np.zeros((64, 64), dtype=np.float32)
            kmax = 0.5

        rows = []
        headers = ["#", "|k| (nm⁻¹)", "d (nm)", "∠(deg)"]
        for r in range(self.table.rowCount()):
            one_row = []
            for c in range(min(self.table.columnCount(), 4)):
                it = self.table.item(r, c)
                one_row.append(it.text() if it is not None else "")
            rows.append(one_row)

        fig = plt.figure(figsize=(6.0, 4.0), constrained_layout=True)
        # 6-column grid: top row uses 2+2+2; bottom row uses 3+3 (1.5 + 1.5 columns).
        gs = fig.add_gridspec(2, 6, height_ratios=[1.0, 1.0], width_ratios=[1.0] * 6)
        ax_img = fig.add_subplot(gs[0, 0:2])
        ax_zoom = fig.add_subplot(gs[0, 2:4])
        ax_fft = fig.add_subplot(gs[0, 4:6])
        ax_profile = fig.add_subplot(gs[1, 0:3])
        ax_table = fig.add_subplot(gs[1, 3:6])

        try:
            def set_square_ticks(ax, xmin, xmax, ymin, ymax, invert_y=False, nticks=5):
                cx = 0.5 * (xmin + xmax)
                cy = 0.5 * (ymin + ymax)
                span = max(float(xmax - xmin), float(ymax - ymin), 1e-12)
                x0, x1 = cx - 0.5 * span, cx + 0.5 * span
                y0, y1 = cy - 0.5 * span, cy + 0.5 * span
                ax.set_xlim(x0, x1)
                if invert_y:
                    ax.set_ylim(y1, y0)
                else:
                    ax.set_ylim(y0, y1)

                # Use "nice" major ticks (1/2/5 * 10^n), better for manual scalebars.
                raw_step = span / max(2, nticks - 1)
                if raw_step <= 0:
                    raw_step = 1.0
                p10 = 10.0 ** np.floor(np.log10(raw_step))
                candidates = np.array([1.0, 2.0, 5.0, 10.0]) * p10
                step = float(candidates[np.searchsorted(candidates, raw_step, side="left")])

                t_start = np.ceil(x0 / step) * step
                t_end = np.floor(x1 / step) * step
                if t_end < t_start:
                    ticks = np.array([x0, x1], dtype=float)
                else:
                    n = int(np.floor((t_end - t_start) / step)) + 1
                    ticks = t_start + step * np.arange(max(2, n), dtype=float)

                # Stable label precision based on step size.
                if step >= 1:
                    ticks = np.round(ticks, 0)
                elif step >= 0.1:
                    ticks = np.round(ticks, 1)
                elif step >= 0.01:
                    ticks = np.round(ticks, 2)
                else:
                    ticks = np.round(ticks, 3)

                ticks = np.unique(ticks)
                ax.set_xticks(ticks)
                ax.set_yticks(ticks)

            # 1) image + ROI
            if self.img is not None and self.img.size > 0:
                h, w = self.img.shape[:2]
                img_vmin = float(img_levels[0]) if isinstance(img_levels, (list, tuple, np.ndarray)) and len(img_levels) >= 2 else None
                img_vmax = float(img_levels[1]) if isinstance(img_levels, (list, tuple, np.ndarray)) and len(img_levels) >= 2 else None
                if px_nm > 0:
                    extent = [0.0, w * px_nm, h * px_nm, 0.0]
                    ax_img.imshow(
                        self.img, cmap="gray", origin="upper", extent=extent, interpolation="nearest",
                        vmin=img_vmin, vmax=img_vmax
                    )
                    ax_img.set_xlabel("x (nm)")
                    ax_img.set_ylabel("y (nm)")
                    pos = self.roi.pos()
                    size = self.roi.size()
                    rx = float(pos.x()) * px_nm
                    ry = float(pos.y()) * px_nm
                    rs = float(min(size.x(), size.y())) * px_nm
                    ax_img.add_patch(Rectangle((rx, ry), rs, rs, fill=False, edgecolor="lime"))
                    set_square_ticks(ax_img, 0.0, w * px_nm, 0.0, h * px_nm, invert_y=True, nticks=5)
                else:
                    ax_img.imshow(
                        self.img, cmap="gray", origin="upper", interpolation="nearest",
                        vmin=img_vmin, vmax=img_vmax
                    )
                    ax_img.set_xlabel("x (px)")
                    ax_img.set_ylabel("y (px)")
                    pos = self.roi.pos()
                    size = self.roi.size()
                    rx = float(pos.x())
                    ry = float(pos.y())
                    rs = float(min(size.x(), size.y()))
                    ax_img.add_patch(Rectangle((rx, ry), rs, rs, fill=False, edgecolor="lime"))
                    set_square_ticks(ax_img, 0.0, float(w), 0.0, float(h), invert_y=True, nticks=5)
            ax_img.set_title("Image")
            ax_img.set_aspect("equal")
            ax_img.set_box_aspect(1)

            # 2) ROI zoom (new)
            patch = self.extract_roi_patch()
            if patch is not None and patch.size > 0:
                zlo, zhi = robust_levels(patch, p_low=1.0, p_high=99.5)
                ph, pw = patch.shape[:2]
                if px_nm > 0:
                    zextent = [0.0, pw * px_nm, ph * px_nm, 0.0]
                    ax_zoom.imshow(patch, cmap="gray", origin="upper", extent=zextent, interpolation="nearest", vmin=zlo, vmax=zhi)
                    ax_zoom.set_xlabel("x (nm)")
                    ax_zoom.set_ylabel("y (nm)")
                    set_square_ticks(ax_zoom, 0.0, pw * px_nm, 0.0, ph * px_nm, invert_y=True, nticks=5)
                else:
                    ax_zoom.imshow(patch, cmap="gray", origin="upper", interpolation="nearest", vmin=zlo, vmax=zhi)
                    ax_zoom.set_xlabel("x (px)")
                    ax_zoom.set_ylabel("y (px)")
                    set_square_ticks(ax_zoom, 0.0, float(pw), 0.0, float(ph), invert_y=True, nticks=5)
                if self._profile_mode_image:
                    geo = self._line_profile_rect_geometry()
                    if geo is not None:
                        corners = geo["corners"]
                        poly = np.vstack([corners, corners[0]])
                        if px_nm > 0:
                            ax_zoom.plot(poly[:, 0] * px_nm, poly[:, 1] * px_nm, color=(1.0, 0.63, 0.16))
                        else:
                            ax_zoom.plot(poly[:, 0], poly[:, 1], color=(1.0, 0.63, 0.16))
            ax_zoom.set_title("ROI Zoom")
            ax_zoom.set_aspect("equal")
            ax_zoom.set_box_aspect(1)

            # 3) FFT
            fft_vmin = float(fft_levels[0]) if isinstance(fft_levels, (list, tuple, np.ndarray)) and len(fft_levels) >= 2 else None
            fft_vmax = float(fft_levels[1]) if isinstance(fft_levels, (list, tuple, np.ndarray)) and len(fft_levels) >= 2 else None
            ax_fft.imshow(
                fft_mag, cmap="gray", origin="upper",
                extent=[-kmax, kmax, kmax, -kmax], interpolation="nearest",
                vmin=fft_vmin, vmax=fft_vmax
            )
            set_square_ticks(ax_fft, -kmax, kmax, -kmax, kmax, invert_y=True, nticks=5)
            if self.points:
                xs = [float(p["kx"]) for p in self.points]
                ys = [float(p["ky"]) for p in self.points]
                ax_fft.scatter(xs, ys, s=26, facecolors="none", edgecolors="red")
                dx = 0.030 * max(kmax, 1e-6)
                dy = 0.030 * max(kmax, 1e-6)
                for i, (x0, y0) in enumerate(zip(xs, ys), start=1):
                    ax_fft.text(x0 + dx, y0 + dy, str(i), color="red", ha="left", va="bottom")
            if (not self._profile_mode_image) and self._ring_radius is not None and self._ring_radius > 0:
                r = float(np.clip(self._ring_radius, 0.0, max(kmax, 1e-12)))
                circ = plt.Circle((0.0, 0.0), r, fill=False, edgecolor=(0.31, 0.86, 1.0))
                ax_fft.add_patch(circ)
            ax_fft.set_title("FFT")
            ax_fft.set_xlabel("k_x (nm⁻¹)")
            ax_fft.set_ylabel("k_y (nm⁻¹)")
            ax_fft.set_aspect("equal")

            # 4) radial profile
            if self._profile_mode_image:
                x = np.asarray(self._line_profile_x, dtype=np.float32)
                y = np.asarray(self._line_profile_y, dtype=np.float32)
            else:
                x = np.asarray(self._radial_k, dtype=np.float32)
                y = np.asarray(self._radial_y, dtype=np.float32)
            if x.size > 0 and y.size > 0:
                if self.chk_logy.isChecked():
                    y_plot = np.maximum(y, 1e-12)
                    ax_profile.semilogy(x, y_plot)
                else:
                    ax_profile.plot(x, y)
            if (not self._profile_mode_image) and self._ring_radius is not None and self._ring_radius > 0:
                ax_profile.axvline(float(self._ring_radius), color=(1.0, 0.55, 0.0), linestyle="-")
            ax_profile.set_title("ROI Line Profile" if self._profile_mode_image else "Profile")
            if self._profile_mode_image:
                ax_profile.set_xlabel("Distance (nm)" if px_nm > 0 else "Distance (px)")
            else:
                ax_profile.set_xlabel("|k| (nm⁻¹)")
            ax_profile.set_ylabel("Intensity")

            # 5) table
            ax_table.set_title("Table")
            ax_table.axis("off")
            if rows:
                tb = ax_table.table(
                    cellText=rows,
                    colLabels=headers,
                    cellLoc="center",
                    loc="center",
                    bbox=[0.01, 0.01, 0.98, 0.88],
                )
        except Exception as e:
            plt.close(fig)
            QtWidgets.QMessageBox.critical(self, "Save failed", f"Failed to render report:\n{e}")
            return

        try:
            fig.savefig(out_path, format="svg", dpi=dpi)
        except Exception as e:
            plt.close(fig)
            QtWidgets.QMessageBox.critical(self, "Save failed", f"Failed to save SVG:\n{e}")
            return
        finally:
            plt.close(fig)

        try:
            self._inject_svg_metadata(Path(out_path), self._metadata_dict())
        except Exception as e:
            QtWidgets.QMessageBox.warning(
                self,
                "Metadata warning",
                f"SVG saved, but failed to write metadata:\n{e}",
            )

        QtWidgets.QMessageBox.information(self, "Saved", f"SVG saved to:\n{out_path}")

    # -------------------------
    # ROI control
    # -------------------------
    def on_bin2_changed(self, state):
        if state == QtCore.Qt.Checked and self.chk_bin4.isChecked():
            self.chk_bin4.blockSignals(True)
            self.chk_bin4.setChecked(False)
            self.chk_bin4.blockSignals(False)
        self.update_fft()

    def on_bin4_changed(self, state):
        if state == QtCore.Qt.Checked and self.chk_bin2.isChecked():
            self.chk_bin2.blockSignals(True)
            self.chk_bin2.setChecked(False)
            self.chk_bin2.blockSignals(False)
        self.update_fft()

    def set_roi_size(self, val: int):
        self.roi_size = int(val)
        if self.img is None:
            return

        pos = self.roi.pos()
        size = self.roi.size()
        cx = pos.x() + size.x() / 2.0
        cy = pos.y() + size.y() / 2.0

        h, w = self.img.shape[:2]
        s = int(min(self.roi_size, h, w))
        s = (s // 2) * 2  # even
        x0 = cx - s / 2.0
        y0 = cy - s / 2.0
        x0, y0, s = _clamp_roi_square(x0, y0, s, w, h)

        self.roi.blockSignals(True)
        self.roi.setPos([x0, y0])
        self.roi.setSize([s, s])
        self._apply_roi_angle()
        self.roi.blockSignals(False)

        self.update_fft()

    def _apply_roi_angle(self):
        try:
            # Rotate around ROI center using normalized local center.
            self.roi.setAngle(float(self.roi_angle_deg), center=(0.5, 0.5))
        except Exception:
            try:
                self.roi.setAngle(float(self.roi_angle_deg))
            except Exception:
                pass

    def set_roi_angle(self, val: float):
        self.roi_angle_deg = float(val)
        self.roi.blockSignals(True)
        self._apply_roi_angle()
        self.roi.blockSignals(False)
        self.update_fft()

    def on_roi_changed(self):
        if self.img is None:
            return

        size = self.roi.size()
        s = float(min(size.x(), size.y()))
        s = float((int(s) // 2) * 2)  # keep even

        pos = self.roi.pos()
        x0 = float(pos.x())
        y0 = float(pos.y())

        h, w = self.img.shape[:2]
        x0, y0, s = _clamp_roi_square(x0, y0, s, w, h)

        self.roi.blockSignals(True)
        self.roi.setPos([x0, y0])
        self.roi.setSize([s, s])
        self._apply_roi_angle()
        self.roi.blockSignals(False)

        self.update_fft()

    # -------------------------
    # ROI extraction
    # -------------------------
    def extract_roi_patch(self):
        if self.img is None:
            return None
        # Use pyqtgraph's own ROI sampling transform so extracted ROI matches
        # exactly what is shown by the rotated ROI overlay on the image view.
        try:
            patch = self.roi.getArrayRegion(self.img, self.img_item, axes=(0, 1))
        except Exception:
            return None

        if patch is None:
            return None
        patch = np.asarray(patch, dtype=np.float32)
        if patch.ndim > 2:
            patch = patch.reshape((-1,) + patch.shape[-2:])[0]
        if patch.size == 0:
            return None

        # Keep square/even behavior expected by downstream FFT pipeline.
        ss = int(min(patch.shape[0], patch.shape[1]))
        ss = (ss // 2) * 2
        if ss < 2:
            return None
        patch = patch[:ss, :ss]
        patch = np.nan_to_num(patch, nan=0.0, posinf=0.0, neginf=0.0)
        return patch

    # -------------------------
    # FFT update (optional binning + k-space axes)
    # -------------------------
    def update_fft(self):
        if self.img is None:
            z = np.zeros((256, 256), dtype=np.float32)
            self.fft_item.setImage(z, autoLevels=False)
            self.fft_item.setLevels([0.0, 1.0])
            self._fft_kmax = 0.5
            self._fft_n = 256
            self.set_fft_center_lines(0.0, 0.0)
            self._radial_k = np.zeros(0, dtype=np.float32)
            self._radial_y = np.zeros(0, dtype=np.float32)
            self.update_radial_plot()
            self.update_zoom_view()
            return

        patch = self.extract_roi_patch()
        if patch is None or patch.size == 0 or patch.shape[0] < 8:
            self.update_zoom_view()
            return

        px_nm = float(self.edit_px.value())  # nm/px (read-only display)
        bin_factor = 1
        if hasattr(self, 'chk_bin4') and self.chk_bin4.isChecked():
            bin_factor = 4
        elif hasattr(self, 'chk_bin2') and self.chk_bin2.isChecked():
            bin_factor = 2
        mag, extent_k, _px_nm_eff, n = fft_mag_log_with_axes(patch, px_nm=px_nm, bin_factor=bin_factor)

        # Contrast handling
        self.fft_item.setImage(mag, autoLevels=False)

        auto_c = True
        if hasattr(self, 'chk_autocontrast'):
            auto_c = bool(self.chk_autocontrast.isChecked())

        if auto_c or self._fft_levels is None:
            lo, hi = robust_levels(mag, p_low=5.0, p_high=99.8)
            self._fft_levels = (lo, hi)
        else:
            lo, hi = self._fft_levels

        self.fft_item.setLevels([lo, hi])

        # k-space axis range (nm^-1)
        kmax = float(extent_k[1])

        # IMPORTANT: Map the ImageItem's pixel coordinates onto physical k-space extents.
        # Without this, the image stays in 0..N pixel coordinates and appears off-center.
        self.fft_item.setRect(QtCore.QRectF(-kmax, -kmax, 2.0 * kmax, 2.0 * kmax))

        # Lock view to the same k-space range
        self.fft_plot.setXRange(-kmax, kmax, padding=0.0)
        self.fft_plot.setYRange(-kmax, kmax, padding=0.0)

        # center lines at k=0
        self.set_fft_center_lines(0.0, 0.0)

        # store for click validity
        self._fft_kmax = kmax
        self._fft_n = int(n)
        self.update_ring_on_fft()

        # radial profile from FFT image
        r_px, y_mean = radial_profile_mean(mag)
        if r_px.size > 0 and n > 0:
            self._radial_k = (r_px * (2.0 * kmax / float(n))).astype(np.float32)
            self._radial_y = y_mean.astype(np.float32)
        else:
            self._radial_k = np.zeros(0, dtype=np.float32)
            self._radial_y = np.zeros(0, dtype=np.float32)
        self.update_radial_plot()
        self.update_zoom_view()

        # clear picks because mapping changed with different ROI size
        # (optional) comment out if you want to preserve picks visually
        # self.clear_picks()

    def set_fft_center_lines(self, cx, cy):
        self.vline.setPos(cx)
        self.hline.setPos(cy)

    def update_radial_plot(self):
        if self._profile_mode_image:
            x = self._line_profile_x
            y = np.asarray(self._line_profile_y, dtype=np.float32)
            px_nm = float(self.edit_px.value())
            self.radial_plot.setTitle("ROI Line Profile")
            self.radial_plot.setLabel('bottom', "Distance (nm)" if px_nm > 0 else "Distance (px)")
            self.radial_plot.setLabel('left', "Mean Intensity")
        else:
            x = self._radial_k
            y = np.asarray(self._radial_y, dtype=np.float32)
            self.radial_plot.setTitle("Radial Profile")
            self.radial_plot.setLabel('bottom', '|k| (nm⁻¹)')
            self.radial_plot.setLabel('left', 'Mean FFT Intensity')

        if x.size == 0 or y.size == 0:
            self.radial_curve.setData([], [])
            if self._radial_vline_item is not None:
                self._radial_vline_item.setVisible(False)
            if self._profile_readout_item is not None:
                self._profile_readout_item.setVisible(False)
            if self._line_profile_click_marker is not None:
                self._line_profile_click_marker.setVisible(False)
            if self._profile_region_item is not None:
                self._profile_region_item.setVisible(False)
            return

        logy = bool(self.chk_logy.isChecked()) if hasattr(self, 'chk_logy') else False
        if logy:
            y = np.maximum(y, 1e-12)
            self.radial_plot.getPlotItem().setLogMode(x=False, y=True)
        else:
            self.radial_plot.getPlotItem().setLogMode(x=False, y=False)
        self.radial_curve.setData(x, y)
        if self._profile_mode_image:
            if self._radial_vline_item is not None:
                self._radial_vline_item.setVisible(False)
            if self._profile_region_item is not None:
                xmin = float(np.min(x))
                xmax = float(np.max(x))
                if np.isfinite(xmin) and np.isfinite(xmax) and xmax > xmin:
                    r0, r1 = self._coerce_profile_range(xmin, xmax)
                    self._profile_region_item.blockSignals(True)
                    self._profile_region_item.setBounds([xmin, xmax])
                    self._profile_region_item.setRegion([r0, r1])
                    self._profile_region_item.blockSignals(False)
                    self._profile_region_item.setVisible(True)
                    self._profile_range = (r0, r1)
                    self._update_profile_range_readout()
                else:
                    self._profile_region_item.setVisible(False)
        elif self._ring_radius is not None and self._radial_vline_item is not None:
            if self._profile_region_item is not None:
                self._profile_region_item.setVisible(False)
            self._radial_vline_item.setPos(float(self._ring_radius))
            self._radial_vline_item.setVisible(True)
        elif self._radial_vline_item is not None:
            if self._profile_region_item is not None:
                self._profile_region_item.setVisible(False)
            self._radial_vline_item.setVisible(False)

        self._update_profile_readout_position()

    def _coerce_profile_range(self, xmin: float, xmax: float):
        if not np.isfinite(xmin) or not np.isfinite(xmax):
            return 0.0, 1.0
        if xmax <= xmin:
            return xmin, xmax
        if isinstance(self._profile_range, (list, tuple)) and len(self._profile_range) == 2:
            r0 = float(np.clip(min(self._profile_range[0], self._profile_range[1]), xmin, xmax))
            r1 = float(np.clip(max(self._profile_range[0], self._profile_range[1]), xmin, xmax))
        elif self._profile_click_x is not None:
            xc = float(np.clip(self._profile_click_x, xmin, xmax))
            span = max((xmax - xmin) * 0.1, 1e-9)
            r0 = max(xmin, xc - 0.5 * span)
            r1 = min(xmax, xc + 0.5 * span)
            self._profile_click_x = None
        else:
            r0, r1 = xmin, xmax
        if r1 <= r0:
            eps = max((xmax - xmin) * 1e-3, 1e-9)
            r1 = min(xmax, r0 + eps)
            if r1 <= r0:
                r0 = max(xmin, xmax - eps)
                r1 = xmax
        return r0, r1

    def on_profile_region_changed(self):
        if self._profile_region_item is None or not self._profile_mode_image:
            return
        try:
            r0, r1 = self._profile_region_item.getRegion()
        except Exception:
            return
        if not np.isfinite(r0) or not np.isfinite(r1):
            return
        a = float(r0)
        b = float(r1)
        r0 = float(min(a, b))
        r1 = float(max(a, b))
        self._profile_range = (r0, r1)
        self._update_profile_range_readout()
        self._update_line_profile_click_marker()

    def _update_profile_range_readout(self):
        if (not self._profile_mode_image) or self._profile_readout_item is None:
            return
        if not (isinstance(self._profile_range, (list, tuple)) and len(self._profile_range) == 2):
            self._profile_readout_item.setVisible(False)
            return
        r0 = float(self._profile_range[0])
        r1 = float(self._profile_range[1])
        if not np.isfinite(r0) or not np.isfinite(r1) or r1 <= r0:
            self._profile_readout_item.setVisible(False)
            return
        unit = "nm" if float(self.edit_px.value()) > 0 else "px"
        self._profile_readout_item.setText(f"{(r1 - r0):.4f} {unit}")
        self._profile_readout_item.setVisible(True)
        self._update_profile_readout_position()

    def _update_profile_readout_position(self):
        if self._profile_readout_item is None or (not self._profile_readout_item.isVisible()):
            return
        vr = self.radial_plot.getPlotItem().vb.viewRange()
        if vr is None or len(vr) < 2:
            return
        x0 = float(vr[0][0])
        x1 = float(vr[0][1])
        y0 = float(vr[1][0])
        y1 = float(vr[1][1])
        xc = 0.5 * (x0 + x1)
        yc = 0.5 * (y0 + y1)
        if np.isfinite(xc) and np.isfinite(yc):
            self._profile_readout_item.setPos(xc, yc)

    def _update_line_profile_click_marker(self):
        if (not self._profile_mode_image) or self.img is None:
            self._line_profile_click_marker.setVisible(False)
            return

        geo = self._line_profile_rect_geometry()
        if geo is None:
            self._line_profile_click_marker.setVisible(False)
            return

        x1, y1 = geo["p1"]
        x2, y2 = geo["p2"]
        dx = float(x2 - x1)
        dy = float(y2 - y1)
        L = float(np.hypot(dx, dy))
        if L <= 1e-9:
            self._line_profile_click_marker.setVisible(False)
            return

        if not (isinstance(self._profile_range, (list, tuple)) and len(self._profile_range) == 2):
            self._line_profile_click_marker.setVisible(False)
            return
        px_nm = float(self.edit_px.value())
        s_vals = [float(self._profile_range[0]), float(self._profile_range[1])]
        if px_nm > 0:
            s_vals = [s / px_nm for s in s_vals]
        s_vals = [float(np.clip(s, 0.0, L)) for s in s_vals]

        nx = -dy / L
        ny = dx / L
        hw = 0.5 * float(geo.get("width", 1.0))
        xs = []
        ys = []
        for i, s in enumerate(s_vals):
            t = s / L
            x_local = float(x1 + dx * t)
            y_local = float(y1 + dy * t)
            xa = x_local - nx * hw
            ya = y_local - ny * hw
            xb = x_local + nx * hw
            yb = y_local + ny * hw
            if i > 0:
                xs.append(np.nan)
                ys.append(np.nan)
            xs.extend([xa, xb])
            ys.extend([ya, yb])
        self._line_profile_click_marker.setData(xs, ys)
        self._line_profile_click_marker.setVisible(True)

    def update_ring_on_fft(self):
        if self._profile_mode_image or self._ring_radius is None or self._fft_kmax is None:
            self.ring_item.setVisible(False)
            if self._radial_vline_item is not None:
                self._radial_vline_item.setVisible(False)
            return
        r = float(np.clip(self._ring_radius, 0.0, float(self._fft_kmax)))
        self.ring_item.setRect(QtCore.QRectF(-r, -r, 2.0 * r, 2.0 * r))
        self.ring_item.setVisible(r > 0)
        if self._radial_vline_item is not None:
            self._radial_vline_item.setPos(r)
            self._radial_vline_item.setVisible(r > 0)

    def on_radial_clicked(self, event):
        if event.button() != QtCore.Qt.LeftButton:
            return
        scene_pos = event.scenePos()
        if not self.radial_plot.getPlotItem().sceneBoundingRect().contains(scene_pos):
            return

        vb = self.radial_plot.getPlotItem().vb
        mp = vb.mapSceneToView(scene_pos)
        x_click = float(mp.x())
        if not np.isfinite(x_click):
            return
        if self._profile_mode_image:
            # In ROI line-profile mode, distance selection is done by drag-selecting range.
            return

        if self._fft_kmax is None:
            return
        r = float(np.clip(abs(x_click), 0.0, float(self._fft_kmax)))
        self._ring_radius = r
        self.update_ring_on_fft()
        d = np.inf if r <= 1e-12 else (1.0 / r)
        d_text = "inf" if not np.isfinite(d) else f"{d:.4f}"
        self._profile_readout_item.setText(f"|k| = {r:.4f} nm⁻¹\nd = {d_text} nm")
        self._profile_readout_item.setVisible(True)
        self._update_profile_readout_position()
        self.update_radial_plot()

    def update_zoom_view(self):
        patch = self.extract_roi_patch()
        if patch is None or patch.size == 0:
            self.zoom_item.setImage(np.zeros((32, 32), dtype=np.float32), autoLevels=False)
            self.zoom_item.setLevels([0.0, 1.0])
            if self._line_profile_roi is not None:
                self._line_profile_roi.setVisible(False)
            if self._line_profile_box_item is not None:
                self._line_profile_box_item.setVisible(False)
            if self._line_profile_click_marker is not None:
                self._line_profile_click_marker.setVisible(False)
            self._line_profile_x = np.zeros(0, dtype=np.float32)
            self._line_profile_y = np.zeros(0, dtype=np.float32)
            return

        lo, hi = robust_levels(patch, p_low=1.0, p_high=99.5)
        self.zoom_item.setImage(patch, autoLevels=False)
        self.zoom_item.setLevels([lo, hi])
        self._ensure_line_profile_roi(patch.shape)
        if self._line_profile_roi is not None:
            self._line_profile_roi.setVisible(self._profile_mode_image)
        self.update_line_profile_box_overlay()

        px_nm = float(self.edit_px.value())
        if px_nm > 0:
            self.zoom_plot.setLabel('bottom', 'x (nm)')
            self.zoom_plot.setLabel('left', 'y (nm)')
            self.zoom_plot.getAxis('bottom').setScale(px_nm)
            self.zoom_plot.getAxis('left').setScale(px_nm)
        else:
            self.zoom_plot.setLabel('bottom', 'x (px)')
            self.zoom_plot.setLabel('left', 'y (px)')
            self.zoom_plot.getAxis('bottom').setScale(1.0)
            self.zoom_plot.getAxis('left').setScale(1.0)

        if self._profile_mode_image:
            self.update_line_profile_from_roi()

    def on_profile_mode_changed(self, _state):
        self._profile_mode_image = bool(self.chk_image_profile.isChecked())
        self._profile_range = None
        self._profile_click_x = None
        if self._profile_readout_item is not None:
            self._profile_readout_item.setVisible(False)
        if self._line_profile_click_marker is not None:
            self._line_profile_click_marker.setVisible(False)
        if self._profile_region_item is not None:
            self._profile_region_item.setVisible(False)
        if self._profile_mode_image:
            self.ring_item.setVisible(False)
            if self._radial_vline_item is not None:
                self._radial_vline_item.setVisible(False)
        else:
            self.update_ring_on_fft()
        self.update_zoom_view()
        self.update_line_profile_box_overlay()
        self.update_radial_plot()

    def on_line_profile_width_changed(self, _val: int):
        self.update_line_profile_box_overlay()
        if self._profile_mode_image:
            self.update_line_profile_from_roi()

    def _ensure_line_profile_roi(self, patch_shape):
        h, w = int(patch_shape[0]), int(patch_shape[1])
        if h < 2 or w < 2:
            return
        need_create = self._line_profile_roi is None
        if not need_create and hasattr(self, "_line_profile_shape"):
            prev_h, prev_w = self._line_profile_shape
            need_create = (prev_h != h) or (prev_w != w)

        if need_create:
            if self._line_profile_roi is not None:
                try:
                    self.zoom_plot.removeItem(self._line_profile_roi)
                except Exception:
                    pass
            x1, y1 = 0.15 * w, 0.5 * h
            x2, y2 = 0.85 * w, 0.5 * h
            self._line_profile_roi = pg.LineSegmentROI(
                [[x1, y1], [x2, y2]],
                pen=pg.mkPen(255, 160, 40, width=1.2),
                movable=True
            )
            self.zoom_plot.addItem(self._line_profile_roi)
            self._line_profile_roi.sigRegionChanged.connect(self.on_line_profile_roi_changed)
            self._line_profile_shape = (h, w)
            self.update_line_profile_box_overlay()

    def on_line_profile_roi_changed(self):
        self.update_line_profile_box_overlay()
        if self._profile_mode_image:
            self.update_line_profile_from_roi()

    def _line_profile_endpoints(self):
        if self._line_profile_roi is None:
            return None
        try:
            shp = self._line_profile_roi.getSceneHandlePositions()
            if len(shp) < 2:
                return None
            p1 = self.zoom_plot.vb.mapSceneToView(shp[0][1])
            p2 = self.zoom_plot.vb.mapSceneToView(shp[1][1])
            return float(p1.x()), float(p1.y()), float(p2.x()), float(p2.y())
        except Exception:
            return None

    def _line_profile_rect_geometry(self):
        pts = self._line_profile_endpoints()
        if pts is None:
            return None
        x1, y1, x2, y2 = pts
        dx = x2 - x1
        dy = y2 - y1
        L = float(np.hypot(dx, dy))
        if L <= 1e-6:
            return None
        width_px = float(max(1, int(self.spin_profile_width.value()))) if hasattr(self, "spin_profile_width") else 1.0
        nx = -dy / L
        ny = dx / L
        hw = 0.5 * width_px
        c1 = (x1 + nx * hw, y1 + ny * hw)
        c2 = (x2 + nx * hw, y2 + ny * hw)
        c3 = (x2 - nx * hw, y2 - ny * hw)
        c4 = (x1 - nx * hw, y1 - ny * hw)
        corners = np.array([c1, c2, c3, c4], dtype=np.float32)
        return {
            "p1": (x1, y1),
            "p2": (x2, y2),
            "width": width_px,
            "length": L,
            "corners": corners,
        }

    def update_line_profile_box_overlay(self):
        if self._line_profile_box_item is None:
            return
        geo = self._line_profile_rect_geometry()
        if geo is None or (not self._profile_mode_image):
            self._line_profile_box_item.setVisible(False)
            return
        corners = geo["corners"]
        poly = np.vstack([corners, corners[0]])
        self._line_profile_box_item.setData(poly[:, 0], poly[:, 1])
        self._line_profile_box_item.setVisible(True)

    @staticmethod
    def _bilinear_sample(img: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        x0 = np.floor(xs).astype(np.int64)
        y0 = np.floor(ys).astype(np.int64)
        x1 = x0 + 1
        y1 = y0 + 1

        valid = (x0 >= 0) & (x1 < w) & (y0 >= 0) & (y1 < h)
        out = np.full(xs.shape, np.nan, dtype=np.float32)
        if not np.any(valid):
            return out

        xv = xs[valid]
        yv = ys[valid]
        x0v = x0[valid]
        y0v = y0[valid]
        x1v = x1[valid]
        y1v = y1[valid]

        Ia = img[y0v, x0v].astype(np.float32)
        Ib = img[y1v, x0v].astype(np.float32)
        Ic = img[y0v, x1v].astype(np.float32)
        Id = img[y1v, x1v].astype(np.float32)

        wa = (x1v - xv) * (y1v - yv)
        wb = (x1v - xv) * (yv - y0v)
        wc = (xv - x0v) * (y1v - yv)
        wd = (xv - x0v) * (yv - y0v)
        out[valid] = Ia * wa + Ib * wb + Ic * wc + Id * wd
        return out

    def update_line_profile_from_roi(self):
        patch = self.extract_roi_patch()
        geo = self._line_profile_rect_geometry()
        if patch is None or patch.size == 0 or geo is None:
            self._line_profile_x = np.zeros(0, dtype=np.float32)
            self._line_profile_y = np.zeros(0, dtype=np.float32)
            self.update_radial_plot()
            self._update_line_profile_click_marker()
            return

        (x1, y1) = geo["p1"]
        (x2, y2) = geo["p2"]
        dx = x2 - x1
        dy = y2 - y1
        L = float(np.hypot(dx, dy))
        if L <= 1e-6:
            self._line_profile_x = np.zeros(0, dtype=np.float32)
            self._line_profile_y = np.zeros(0, dtype=np.float32)
            self.update_radial_plot()
            self._update_line_profile_click_marker()
            return

        ns = int(max(32, np.ceil(L)))
        t = np.linspace(0.0, 1.0, ns, dtype=np.float32)
        cx = x1 + dx * t
        cy = y1 + dy * t

        nx = -dy / L
        ny = dx / L
        width_px = int(max(1.0, float(geo["width"])))
        offs = np.linspace(-(width_px - 1) * 0.5, (width_px - 1) * 0.5, width_px, dtype=np.float32)

        stack = []
        for o in offs:
            xs = cx + nx * o
            ys = cy + ny * o
            stack.append(self._bilinear_sample(patch, xs, ys))
        arr = np.vstack(stack)
        prof = np.nanmean(arr, axis=0).astype(np.float32)
        dist = np.linspace(0.0, L, ns, dtype=np.float32)

        px_nm = float(self.edit_px.value())
        if px_nm > 0:
            dist = dist * px_nm

        valid = np.isfinite(prof)
        self._line_profile_x = dist[valid]
        self._line_profile_y = prof[valid]
        self.update_radial_plot()
        self._update_line_profile_click_marker()

    # -------------------------
    # Picking on FFT (k-space)
    # -------------------------
    @staticmethod
    def _click_hits_viewbox_overlay(vb, scene_pos) -> bool:
        for name in ("autoBtn", "menuBtn"):
            item = getattr(vb, name, None)
            if item is None:
                continue
            try:
                if item.isVisible() and item.sceneBoundingRect().contains(scene_pos):
                    return True
            except Exception:
                continue
        return False

    def on_fft_clicked(self, event):
        if self.img is None:
            return
        if event.button() != QtCore.Qt.LeftButton:
            return
        if self._fft_kmax is None:
            return

        vb = self.fft_plot.vb
        scene_pos = event.scenePos()
        # Only clicks in the ViewBox data area are valid (exclude axes/ticks/margins).
        if not vb.sceneBoundingRect().contains(scene_pos):
            return
        if self._click_hits_viewbox_overlay(vb, scene_pos):
            return

        mp = vb.mapSceneToView(scene_pos)
        kx = float(mp.x())   # nm^-1
        ky = float(mp.y())   # nm^-1

        kmax = float(self._fft_kmax)
        if not (-kmax <= kx <= kmax and -kmax <= ky <= kmax):
            return

        k = float(np.hypot(kx, ky))
        if k <= 1e-12:
            return
        d = 1.0 / k

        pick = {"kx": kx, "ky": ky, "k": k, "d": d}
        # angle relative to the first pick (about origin)
        if len(self.points) == 0:
            pick['angle_deg'] = 0.0
        else:
            pick['angle_deg'] = self._angle_to_first_deg(kx, ky)
        self.points.append(pick)

        # update scatter
        self.update_scatter()

        # add numbered label near the point
        idx = len(self.points)
        t = pg.TextItem(text=str(idx), anchor=(0, 1), color=(255, 80, 80))
        t.setPos(kx, ky)
        self.fft_plot.addItem(t)
        self.text_labels.append(t)

        # add to table
        self.append_table_row(pick)

    def update_scatter(self):
        spots = [{"pos": (p["kx"], p["ky"])} for p in self.points]
        self.scatter.setData(spots)

    def _angle_to_first_deg(self, kx: float, ky: float) -> float:
        """Angle (degrees) between vector (kx,ky) and the first pick vector, about the origin."""
        if not self.points:
            return float('nan')
        kx0 = float(self.points[0].get('kx', 0.0))
        ky0 = float(self.points[0].get('ky', 0.0))
        # norms
        n0 = float(np.hypot(kx0, ky0))
        n1 = float(np.hypot(kx, ky))
        if n0 <= 1e-12 or n1 <= 1e-12:
            return float('nan')
        # dot -> acos for smallest angle 0..180
        cosang = (kx0 * kx + ky0 * ky) / (n0 * n1)
        cosang = float(np.clip(cosang, -1.0, 1.0))
        ang = float(np.degrees(np.arccos(cosang)))
        return ang

    def refresh_table_all(self):
        """Rebuild the table from current points, recomputing angles relative to #1."""
        self.table.setRowCount(0)
        for i, p in enumerate(self.points):
            # recompute angle
            if i == 0:
                p['angle_deg'] = 0.0
            else:
                p['angle_deg'] = self._angle_to_first_deg(float(p['kx']), float(p['ky']))
            self.append_table_row(p)

    # -------------------------
    # Table
    # -------------------------
    def configure_table_columns(self):
        """Set compact, fixed table widths matching numeric formats."""
        hh = self.table.horizontalHeader()
        hh.setStretchLastSection(False)
        hh.setSectionResizeMode(QtWidgets.QHeaderView.Fixed)

        fm = self.table.fontMetrics()
        pad = 18

        w_idx = fm.horizontalAdvance("99") + pad
        w_k = max(fm.horizontalAdvance("|k| (nm⁻¹)"), fm.horizontalAdvance("000.0000")) + pad
        w_d = max(fm.horizontalAdvance("d (nm)"), fm.horizontalAdvance("000.0000")) + pad
        w_ang = max(fm.horizontalAdvance("∠(deg)"), fm.horizontalAdvance("180.00")) + pad

        self.table.setColumnWidth(0, w_idx)
        self.table.setColumnWidth(1, w_k)
        self.table.setColumnWidth(2, w_d)
        self.table.setColumnWidth(3, w_ang)

    def append_table_row(self, pick):
        row = self.table.rowCount()
        self.table.insertRow(row)

        ang = pick.get('angle_deg', float('nan'))
        ang_str = f"{ang:.2f}" if np.isfinite(ang) else "—"
        vals = [
            str(row + 1),
            f"{pick['k']:.4f}",
            f"{pick['d']:.4f}",
            ang_str,
        ]
        for c, v in enumerate(vals):
            item = QtWidgets.QTableWidgetItem(v)
            item.setTextAlignment(QtCore.Qt.AlignCenter)
            self.table.setItem(row, c, item)

    # -------------------------
    # Picks management
    # -------------------------
    def clear_picks(self):
        self.points = []
        self.scatter.setData([])
        self.table.setRowCount(0)
        self._ring_radius = None
        self.update_ring_on_fft()
        self.update_radial_plot()

        for t in self.text_labels:
            try:
                self.fft_plot.removeItem(t)
            except Exception:
                pass
        self.text_labels = []


    def remove_last_pick(self):
        """Remove the most recently added pick (last point/label/table row)."""
        if not self.points:
            return

        # remove last point
        self.points.pop(-1)

        # remove last label
        if self.text_labels:
            t = self.text_labels.pop(-1)
            try:
                self.fft_plot.removeItem(t)
            except Exception:
                pass

        # update scatter
        self.update_scatter()

        # renumber remaining labels
        for i, t in enumerate(self.text_labels, start=1):
            t.setText(str(i))
            t.setPos(self.points[i - 1]['kx'], self.points[i - 1]['ky'])

        # rebuild table to update angles relative to new #1
        self.refresh_table_all()


def launch_hrtem_analyzer(file_path=None, show=True):
    """
    Launch the HRTEM analyzer window programmatically.
    Returns: (window, app)
    """
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])

    pg.setConfigOptions(antialias=True)
    w = HRTEMFFTAnalyzer()

    if file_path is not None:
        w.edit_load_path.setText(str(file_path))
        w.open_image()

    if show:
        w.show()

    return w, app


def main():
    _w, app = launch_hrtem_analyzer(show=True)
    app.exec_()


if __name__ == "__main__":
    main()
