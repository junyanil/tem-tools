from __future__ import annotations

# Self-contained GitHub release prepared with OpenAI Codex assistance (2026-03-10).
"""
SAED indexing GUI for loading a single diffraction image, calibrating scale,
inspecting azimuthal profiles, and exporting annotated SVG results.
"""

import json
import math
import re
from pathlib import Path

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg

pg.setConfigOptions(imageAxisOrder="row-major")


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


def load_hyperspy_2d(file_path: str | Path):
    """
    Load the first 2D dataset from a hyperspy-compatible file.
    Returns image data and the scale metadata needed by the SAED GUI.
    """
    path = Path(file_path).expanduser()
    datasets = import_data_hyperspy(path.as_posix())
    if not isinstance(datasets, (list, tuple)) or len(datasets) == 0:
        raise ValueError("No datasets were found in the selected file.")

    sub_data = datasets[0]
    axes = sub_data.get("axes", [])
    data_dimension = len(axes)
    if data_dimension == 1:
        metadata_axis = axes[0]
    elif data_dimension in {2, 3}:
        metadata_axis = axes[1]
    else:
        raise ValueError("Unsupported data dimension in hyperspy dataset.")

    return (
        sub_data["data"],
        metadata_axis.get("scale", None),
        metadata_axis.get("units", None),
        metadata_axis.get("size", None),
    )


def clipped_8bit_image(
    image: np.ndarray,
    clipping_percentile: float = 0.3,
    asymetric_clipping_percentile=None,
) -> np.ndarray:
    img = np.asarray(image, dtype=np.float32)
    if asymetric_clipping_percentile is not None:
        low_p, high_p = asymetric_clipping_percentile
        vmin, vmax = np.percentile(img, [low_p, 100.0 - high_p])
    else:
        vmin, vmax = np.percentile(img, [clipping_percentile, 100.0 - clipping_percentile])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin = float(np.min(img))
        vmax = float(np.max(img))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        return np.zeros_like(img, dtype=np.uint8)
    out = np.clip((img - vmin) / (vmax - vmin) * 255.0, 0, 255)
    return out.astype(np.uint8)


def circle_center_from_3pts(p1, p2, p3, eps=1e-9):
    x1, y1 = map(float, p1)
    x2, y2 = map(float, p2)
    x3, y3 = map(float, p3)
    d = 2.0 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    if abs(d) < eps:
        raise ValueError("3 points are nearly collinear.")
    s1 = x1 * x1 + y1 * y1
    s2 = x2 * x2 + y2 * y2
    s3 = x3 * x3 + y3 * y3
    ux = (s1 * (y2 - y3) + s2 * (y3 - y1) + s3 * (y1 - y2)) / d
    uy = (s1 * (x3 - x2) + s2 * (x1 - x3) + s3 * (x2 - x1)) / d
    return float(ux), float(uy)


def _scale_to_nm_inv_per_px(scale, units) -> float:
    if scale is None:
        return 0.0
    try:
        sc = float(scale)
    except Exception:
        return 0.0
    if sc <= 0:
        return 0.0
    u = (str(units).strip().lower() if units is not None else "")
    if u in ("1/nm", "nm^-1", "nm⁻¹"):
        return sc
    if u in ("1/å", "1/a", "a^-1", "å^-1", "1/angstrom", "angstrom^-1"):
        return sc * 10.0
    if u in ("1/pm", "pm^-1"):
        return sc * 1000.0
    if u in ("1/m", "m^-1"):
        return sc * 1e-9
    return sc


def _power_law(x, a, b, c):
    return a * (x ** b) + c


class TwoHandleRangeBar(QtWidgets.QWidget):
    rangeChanged = QtCore.pyqtSignal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(30)
        self._min = 0.0
        self._max = 1.0
        self._left = 0.0
        self._right = 1.0
        self._active = None
        self._radius = 7
        self._margin = 10

    def set_values(self, vmin: float, vmax: float, left: float, right: float):
        self._min = float(vmin)
        self._max = float(max(vmax, vmin + 1e-12))
        l = float(np.clip(left, self._min, self._max))
        r = float(np.clip(right, self._min, self._max))
        if r <= l:
            r = min(self._max, l + max((self._max - self._min) * 1e-3, 1e-9))
        self._left = l
        self._right = r
        self.update()
        self.rangeChanged.emit(self._left, self._right)

    def values(self):
        return self._left, self._right

    def _to_px(self, v: float):
        w = max(1, self.width() - 2 * self._margin)
        t = (v - self._min) / (self._max - self._min) if self._max > self._min else 0.0
        return self._margin + t * w

    def _from_px(self, x: float):
        w = max(1, self.width() - 2 * self._margin)
        t = (x - self._margin) / w
        t = float(np.clip(t, 0.0, 1.0))
        return self._min + t * (self._max - self._min)

    def paintEvent(self, _event):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing, True)
        pal = self.palette()
        c_track = pal.color(QtGui.QPalette.Mid)
        c_sel = pal.color(QtGui.QPalette.Highlight)
        c_handle = pal.color(QtGui.QPalette.Button)
        c_handle_edge = pal.color(QtGui.QPalette.Dark)
        y = self.height() * 0.5
        x0 = self._margin
        x1 = self.width() - self._margin

        p.setPen(QtGui.QPen(c_track, 2))
        p.drawLine(int(x0), int(y), int(x1), int(y))

        xl = self._to_px(self._left)
        xr = self._to_px(self._right)
        p.setPen(QtGui.QPen(c_sel, 3))
        p.drawLine(int(xl), int(y), int(xr), int(y))

        for x in (xl, xr):
            p.setPen(QtGui.QPen(c_handle_edge, 1))
            p.setBrush(QtGui.QBrush(c_handle))
            p.drawEllipse(QtCore.QPointF(x, y), self._radius, self._radius)

    def mousePressEvent(self, event):
        xl = self._to_px(self._left)
        xr = self._to_px(self._right)
        x = float(event.x())
        self._active = "left" if abs(x - xl) <= abs(x - xr) else "right"
        self.mouseMoveEvent(event)

    def mouseMoveEvent(self, event):
        if self._active is None:
            return
        v = self._from_px(float(event.x()))
        if self._active == "left":
            self._left = min(v, self._right - max((self._max - self._min) * 1e-3, 1e-9))
        else:
            self._right = max(v, self._left + max((self._max - self._min) * 1e-3, 1e-9))
        self.update()
        self.rangeChanged.emit(self._left, self._right)

    def mouseReleaseEvent(self, _event):
        self._active = None


class SAEDIndexingWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("SAED Indexer")
        self.resize(1500, 900)

        self.raw_image = None
        self.display_image = None
        self.image_path = None
        self.key = ""
        self.root_path = ""
        self.scale_nm_inv_per_px = 0.0
        self.scale_units = ""
        self.center_xy = None
        self.center_pick_points = []
        self.profile_r = np.zeros(0, dtype=np.float32)
        self.profile_i_raw = np.zeros(0, dtype=np.float32)
        self.profile_bg = np.zeros(0, dtype=np.float32)
        self.profile_sub = np.zeros(0, dtype=np.float32)
        self.picks = []
        self._arc_items = []
        self._vline_items = []
        self._spot_items = []
        self._spot_label_items = []
        self._center_marker = None
        self._center_circle_roi = None
        self._updating_center_circle = False
        self._center_update_timer = QtCore.QTimer(self)
        self._center_update_timer.setSingleShot(True)
        self._center_update_timer.timeout.connect(self._apply_deferred_center_update)
        self._saed_display_rect = None
        self._saed_crop = None
        self._bg_region_items = []
        self._bg_region_low = None
        self._bg_region_high = None
        self._updating_bg_region = False
        self._bg_initialized = False
        self._bg_ylim_initialized = False
        self._plot_x_initialized = False
        self._updating_plot_region = False
        self._plot_auto_peak_pending = True
        self._using_skued = False
        self._active_zone = "pick"
        self._center_method = "ring"      # midpoint | ring | circle
        self._crystal_mode = "poly"       # poly | single
        self._scale_source = "none"
        self._calibrating_scale = False
        self._calib_line_roi = None
        self._updating_calib_line = False
        self._calib_bar_h_px = 4.0
        self._pick_color_hex = [
            "#ff4d4f", "#40a9ff", "#73d13d", "#ffa940",
            "#9254de", "#13c2c2", "#f759ab", "#a0d911",
        ]

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main_layout = QtWidgets.QHBoxLayout(central)
        main_layout.setContentsMargins(8, 8, 8, 8)

        self._build_left_column()
        self._build_middle_column()
        self._build_right_column()

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        splitter.addWidget(self.left_col)
        splitter.addWidget(self.middle_col)
        splitter.addWidget(self.right_col)
        splitter.setSizes([620, 700, 380])
        main_layout.addWidget(splitter)

        self._sc_undo = QtWidgets.QShortcut(QtGui.QKeySequence.Undo, self)
        self._sc_undo.activated.connect(self.undo_contextual)
        self._sc_close = QtWidgets.QShortcut(QtGui.QKeySequence.Close, self)
        self._sc_close.activated.connect(self.close)

    def _build_left_column(self):
        self.left_col = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(self.left_col)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(6)

        self.raw_view = pg.GraphicsLayoutWidget()
        self.raw_plot = self.raw_view.addPlot(row=0, col=0)
        self.raw_plot.setTitle("Raw SAED Preview")
        self.raw_plot.setAspectLocked(True)
        self.raw_plot.invertY(True)
        self.raw_plot.setLabel("bottom", "x (px)")
        self.raw_plot.setLabel("left", "y (px)")
        self.raw_img_item = pg.ImageItem()
        self.raw_plot.addItem(self.raw_img_item)
        self.raw_scatter = pg.ScatterPlotItem(
            size=8,
            pen=pg.mkPen(255, 80, 80, width=1.5),
            brush=pg.mkBrush(255, 80, 80, 120),
        )
        self.raw_plot.addItem(self.raw_scatter)
        self.raw_plot.scene().sigMouseClicked.connect(self.on_raw_clicked)
        left_layout.addWidget(self.raw_view, stretch=7)

        center_row = QtWidgets.QWidget()
        center_row_layout = QtWidgets.QHBoxLayout(center_row)
        center_row_layout.setContentsMargins(0, 0, 0, 0)
        center_row_layout.setSpacing(6)
        self.btn_poly = QtWidgets.QPushButton("Poly crystal")
        self.btn_poly.setCheckable(True)
        self.btn_poly.setChecked(True)
        self.btn_single = QtWidgets.QPushButton("Single crystal")
        self.btn_single.setCheckable(True)
        self._crystal_group = QtWidgets.QButtonGroup(self)
        self._crystal_group.setExclusive(True)
        self._crystal_group.addButton(self.btn_poly)
        self._crystal_group.addButton(self.btn_single)
        self.btn_poly.clicked.connect(self.on_crystal_mode_changed)
        self.btn_single.clicked.connect(self.on_crystal_mode_changed)
        self.btn_method_switch = QtWidgets.QPushButton("Ring(3pt)")
        self.btn_method_switch.clicked.connect(self.on_method_switch)
        self.btn_method_switch.setToolTip("Method Switch")
        self.btn_center_auto = QtWidgets.QPushButton("Auto Center")
        self.btn_center_auto.clicked.connect(self.auto_center)
        center_row_layout.addWidget(self.btn_poly)
        center_row_layout.addWidget(self.btn_single)
        center_row_layout.addWidget(self.btn_method_switch)
        center_row_layout.addWidget(self.btn_center_auto)
        left_layout.addWidget(center_row, stretch=0)

        center_pos_row = QtWidgets.QWidget()
        center_pos_layout = QtWidgets.QHBoxLayout(center_pos_row)
        center_pos_layout.setContentsMargins(0, 0, 0, 0)
        center_pos_layout.setSpacing(6)
        self.lbl_center_xy = QtWidgets.QLabel("Center (px)")
        self.edit_center_xy = QtWidgets.QLineEdit("x=--, y=--")
        self.edit_center_xy.setReadOnly(True)
        center_pos_layout.addWidget(self.lbl_center_xy, stretch=0)
        center_pos_layout.addWidget(self.edit_center_xy, stretch=1)
        left_layout.addWidget(center_pos_row, stretch=0)

        bg_group = QtWidgets.QGroupBox("Background Subtraction")
        bg_layout = QtWidgets.QVBoxLayout(bg_group)
        bg_layout.setContentsMargins(6, 6, 6, 6)
        bg_layout.setSpacing(4)

        self.bg_plot = pg.PlotWidget()
        self.bg_plot.setMinimumHeight(220)
        self.bg_plot.setLabel("bottom", "q (nm^-1)")
        self.bg_plot.setLabel("left", "Intensity")
        self.bg_plot.showGrid(x=True, y=True, alpha=0.2)
        self.bg_curve_raw = self.bg_plot.plot([], [], pen=pg.mkPen(245, 245, 245, width=2.0))
        self.bg_curve_fit = self.bg_plot.plot([], [], pen=pg.mkPen(30, 130, 220, width=2.4))
        self.bg_curve_sub = self.bg_plot.plot([], [], pen=pg.mkPen(220, 80, 80, width=2.0))
        bg_layout.addWidget(self.bg_plot, stretch=1)

        self.lbl_bg = QtWidgets.QLabel("Drag shaded regions to set background fit ranges")
        bg_layout.addWidget(self.lbl_bg)
        left_layout.addWidget(bg_group, stretch=3)

    def _build_middle_column(self):
        self.middle_col = QtWidgets.QWidget()
        mid_layout = QtWidgets.QVBoxLayout(self.middle_col)
        mid_layout.setContentsMargins(0, 0, 0, 0)
        mid_layout.setSpacing(6)

        self.saed_view = pg.GraphicsLayoutWidget()
        self.saed_plot = self.saed_view.addPlot(row=0, col=0)
        self.saed_plot.setAspectLocked(True)
        self.saed_plot.invertY(True)
        self.saed_plot.setTitle("Zoomed SAED")
        self.saed_plot.setLabel("bottom", "k_x (nm^-1)")
        self.saed_plot.setLabel("left", "k_y (nm^-1)")
        self.saed_img_item = pg.ImageItem()
        self.saed_plot.addItem(self.saed_img_item)
        self.saed_cross_v = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen(255, 220, 80, width=1))
        self.saed_cross_h = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen(255, 220, 80, width=1))
        self.saed_plot.addItem(self.saed_cross_v)
        self.saed_plot.addItem(self.saed_cross_h)
        self.saed_cross_v.setVisible(False)
        self.saed_cross_h.setVisible(False)
        self.saed_plot.scene().sigMouseClicked.connect(self.on_saed_clicked)
        mid_layout.addWidget(self.saed_view, stretch=7)

        self.profile_plot = pg.PlotWidget()
        self.profile_plot.setTitle("Rotation Average")
        self.profile_plot.setLabel("bottom", "q (nm^-1)")
        self.profile_plot.setLabel("left", "Intensity")
        self.profile_plot.showGrid(x=True, y=True, alpha=0.2)
        self.profile_curve = self.profile_plot.plot([], [], pen=pg.mkPen(80, 220, 255, width=2.8))
        self.profile_plot.scene().sigMouseClicked.connect(self.on_profile_clicked)
        self._profile_hover = pg.TextItem("", anchor=(1, 0), color=(255, 220, 120))
        self.profile_plot.addItem(self._profile_hover, ignoreBounds=True)
        self._profile_hover.setVisible(False)
        mid_layout.addWidget(self.profile_plot, stretch=3)

    def _build_right_column(self):
        self.right_col = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(self.right_col)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(8)

        io_group = QtWidgets.QGroupBox("Load")
        io_layout = QtWidgets.QGridLayout(io_group)
        self.edit_path = QtWidgets.QLineEdit()
        self.edit_path.setPlaceholderText("Full file path, e.g. /path/to/image.dm4")
        self.btn_browse = QtWidgets.QPushButton("Browse...")
        self.btn_open = QtWidgets.QPushButton("Open SAED")
        self.btn_open_svg = QtWidgets.QPushButton("Open SVG")
        self.btn_save_300 = QtWidgets.QPushButton("Save SVG (300 dpi)")
        self.btn_save_600 = QtWidgets.QPushButton("Save SVG (600 dpi)")
        self.btn_scale = QtWidgets.QPushButton("Calibrate Scale")
        self.edit_scale_display = QtWidgets.QLineEdit("-- nm^-1/px")
        self.edit_scale_display.setReadOnly(True)
        self.btn_open.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.btn_open_svg.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.btn_browse.clicked.connect(self.browse_file)
        self.btn_open.clicked.connect(self.open_image)
        self.btn_open_svg.clicked.connect(self.open_svg)
        self.btn_save_300.clicked.connect(lambda: self.save_svg_with_table(dpi=300))
        self.btn_save_600.clicked.connect(lambda: self.save_svg_with_table(dpi=600))
        self.btn_scale.clicked.connect(self.on_scale_button_clicked)
        self.edit_scale_display.returnPressed.connect(self.on_scale_display_return_pressed)
        self.edit_path.returnPressed.connect(self.open_image)
        row = 0
        io_layout.addWidget(self.btn_browse, row, 0)
        io_layout.addWidget(self.edit_path, row, 1, 1, 2)
        row += 1
        io_layout.setColumnStretch(0, 0)
        io_layout.setColumnStretch(1, 1)
        io_layout.setColumnStretch(2, 0)
        io_layout.addWidget(self.btn_open, row, 0)
        io_layout.addWidget(self.btn_open_svg, row, 1)
        row += 1
        io_layout.addWidget(self.btn_save_300, row, 0)
        io_layout.addWidget(self.btn_save_600, row, 1)
        row += 1
        io_layout.addWidget(self.btn_scale, row, 0)
        io_layout.addWidget(self.edit_scale_display, row, 1)
        right_layout.addWidget(io_group, stretch=0)

        opt_group = QtWidgets.QGroupBox("Options")
        opt_layout = QtWidgets.QGridLayout(opt_group)

        self.btn_clear_pick = QtWidgets.QPushButton("Clear Peaks")
        self.btn_clear_pick.clicked.connect(self.clear_picks)
        self.btn_remove_last = QtWidgets.QPushButton("Remove Last")
        self.btn_remove_last.clicked.connect(self.remove_last_pick)

        self.spin_clip = QtWidgets.QDoubleSpinBox()
        self.spin_clip.setRange(0.1, 50.0)
        self.spin_clip.setSingleStep(0.1)
        self.spin_clip.setDecimals(1)
        self.spin_clip.setValue(0.3)
        self.spin_clip.setSuffix(" %")
        self.spin_clip.valueChanged.connect(self.on_view_option_changed)

        self.chk_log = QtWidgets.QCheckBox("Log intensity")
        self.chk_log.stateChanged.connect(self.on_view_option_changed)

        self.spin_rotation = QtWidgets.QDoubleSpinBox()
        self.spin_rotation.setRange(-180.0, 180.0)
        self.spin_rotation.setDecimals(1)
        self.spin_rotation.setSingleStep(0.5)
        self.spin_rotation.setValue(0.0)
        self.spin_rotation.setSuffix(" deg")
        self.spin_rotation.valueChanged.connect(self.on_view_option_changed)

        self.spin_zoom = QtWidgets.QDoubleSpinBox()
        self.spin_zoom.setRange(0.2, 1.0)
        self.spin_zoom.setSingleStep(0.05)
        self.spin_zoom.setValue(0.6)
        self.spin_zoom.setToolTip("Smaller value means stronger zoom-in.")
        self.spin_zoom.valueChanged.connect(self.refresh_saed_view)

        self.combo_cmap = QtWidgets.QComboBox()
        self.combo_cmap.addItems(["gray", "Greys", "viridis", "plasma", "magma"])
        self.combo_cmap.setCurrentText("gray")
        self.combo_cmap.currentTextChanged.connect(self.on_view_option_changed)

        self.lbl_plot_x = QtWidgets.QLabel("0.00 ~ 0.00")
        self.plot_x_bar = TwoHandleRangeBar()
        self.plot_x_bar.rangeChanged.connect(self.on_plot_range_changed)

        row = 0
        opt_layout.addWidget(self.chk_log, row, 0, 1, 2)
        row += 1
        opt_layout.addWidget(QtWidgets.QLabel("cmap"), row, 0)
        opt_layout.addWidget(self.combo_cmap, row, 1)
        row += 1
        opt_layout.addWidget(QtWidgets.QLabel("Intensity clip"), row, 0)
        opt_layout.addWidget(self.spin_clip, row, 1)
        row += 1
        opt_layout.addWidget(QtWidgets.QLabel("Rotation"), row, 0)
        opt_layout.addWidget(self.spin_rotation, row, 1)
        row += 1
        opt_layout.addWidget(QtWidgets.QLabel("Zoom window ratio"), row, 0)
        opt_layout.addWidget(self.spin_zoom, row, 1)
        row += 1
        opt_layout.addWidget(QtWidgets.QLabel("Range"), row, 0)
        opt_layout.addWidget(self.lbl_plot_x, row, 1)
        row += 1
        opt_layout.addWidget(self.plot_x_bar, row, 0, 1, 2)
        right_layout.addWidget(opt_group, stretch=0)

        self.table = QtWidgets.QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["", "#", "q (nm^-1)", "d (A)", "∠(deg)"])
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QtWidgets.QHeaderView.Stretch)
        header.setSectionResizeMode(3, QtWidgets.QHeaderView.Stretch)
        header.setSectionResizeMode(4, QtWidgets.QHeaderView.ResizeToContents)
        self.table.setColumnWidth(0, 24)

        picks_btn_row = QtWidgets.QWidget()
        picks_btn_layout = QtWidgets.QHBoxLayout(picks_btn_row)
        picks_btn_layout.setContentsMargins(0, 0, 0, 0)
        picks_btn_layout.setSpacing(6)
        picks_btn_layout.addWidget(self.btn_clear_pick)
        picks_btn_layout.addWidget(self.btn_remove_last)
        right_layout.addWidget(picks_btn_row, stretch=0)

        right_layout.addWidget(self.table, stretch=1)

        self.lbl_status = QtWidgets.QLabel("Ready.")
        self.lbl_status.setWordWrap(True)
        right_layout.addWidget(self.lbl_status, stretch=0)

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
        path_text = self.edit_path.text().strip()
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
        self.edit_path.setText(str(selected))
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
        raise ValueError(f"Unsupported image format: {path.suffix}")

    def _raw_preview_title(self) -> str:
        if self.image_path is None:
            return "Raw SAED Preview"
        return Path(self.image_path).expanduser().name

    def _set_scale_nm_inv_per_px(self, value: float, source: str = "none"):
        try:
            sc = float(value)
        except Exception:
            sc = 0.0
        if (not np.isfinite(sc)) or sc <= 0:
            sc = 0.0
        self.scale_nm_inv_per_px = sc
        self.scale_units = "nm^-1"
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
            self.edit_scale_display.setText(f"{sc:.6g} nm^-1/px" if sc > 0 else "-- nm^-1/px")

    def _calibration_line_length_px(self):
        if self._calib_line_roi is None:
            return 0.0
        try:
            return abs(float(self._calib_line_roi.size().x()))
        except Exception:
            return 0.0

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
        if self.display_image is None:
            return
        self._calibrating_scale = True
        self.btn_scale.setText("Apply Scale")
        self.edit_scale_display.setReadOnly(False)
        self.edit_scale_display.clear()
        self.edit_scale_display.setPlaceholderText("nm-1")
        self.edit_scale_display.setFocus()
        if self._calib_line_roi is None:
            h, w = self.display_image.shape[:2]
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
            self.raw_plot.addItem(self._calib_line_roi)
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
            sc = float(self.scale_nm_inv_per_px)
            self.edit_scale_display.setText(f"{sc:.6g} nm^-1/px" if sc > 0 else "-- nm^-1/px")
        self.edit_scale_display.setReadOnly(True)
        if self._scale_source == "metadata" and self.scale_nm_inv_per_px > 0:
            self.btn_scale.setText("Scale (Metadata)")
        elif self._scale_source == "calibrated" and self.scale_nm_inv_per_px > 0:
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
            QtWidgets.QMessageBox.warning(self, "Calibration", "Please input a numeric real length (nm-1).")
            return
        if not np.isfinite(real_len) or real_len <= 0:
            QtWidgets.QMessageBox.warning(self, "Calibration", "Real length must be > 0.")
            return
        px_len = self._calibration_line_length_px()
        if px_len <= 1e-9:
            QtWidgets.QMessageBox.warning(self, "Calibration", "Calibration line length is too small.")
            return
        sc = float(real_len / px_len)
        self._set_scale_nm_inv_per_px(sc, source="calibrated")
        self._exit_scale_calibration_mode(keep_text=False)
        self._plot_x_initialized = False
        self._plot_auto_peak_pending = True
        self.refresh_all()

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
        self.edit_path.setText(str(Path(path).expanduser()))

    def open_image(self):
        try:
            data_path = self._choose_image_file_from_input()
            if data_path is None:
                return
            key = data_path.name
            if self._is_plain_image_suffix(data_path):
                img = self._load_plain_image(data_path)
                scale = 0.0
                units = "nm^-1"
            else:
                img, scale, units, _size = load_hyperspy_2d(data_path)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Open image failed", str(e))
            return

        self._apply_loaded_image(
            img=img,
            data_path=Path(data_path),
            key=key,
            scale=scale,
            units=units,
        )

    def _apply_loaded_image(self, img, data_path: Path, key: str, scale, units):
        img = np.asarray(img)
        if img.ndim != 2:
            QtWidgets.QMessageBox.critical(self, "Open image failed", "Only 2D image is supported.")
            return
        if self._center_update_timer.isActive():
            self._center_update_timer.stop()

        self.raw_image = img.astype(np.float32, copy=False)
        self.display_image = self._rotated_image(self.raw_image)
        self.image_path = Path(data_path)
        self.key = key
        self.root_path = str(data_path.parent)
        self.scale_units = "nm^-1"
        scale_val = _scale_to_nm_inv_per_px(scale, units)
        self.scale_nm_inv_per_px = 0.0
        self._set_scale_nm_inv_per_px(scale_val, source=("metadata" if scale_val > 0 else "none"))
        self.raw_plot.setTitle(self._raw_preview_title())
        self._exit_scale_calibration_mode(keep_text=True)

        self.center_xy = (self.raw_image.shape[1] * 0.5, self.raw_image.shape[0] * 0.5)
        self.center_pick_points.clear()
        self.picks.clear()
        self._clear_overlay_items()
        self._bg_initialized = False
        self._bg_ylim_initialized = False
        self._plot_x_initialized = False
        self._plot_auto_peak_pending = True
        self.refresh_all()
        self.lbl_status.setText(
            f"Loaded: {self.image_path.name}\n"
            f"scale={self.scale_nm_inv_per_px:.6g} nm^-1/px ({self.scale_units})"
        )

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

    def open_svg(self):
        start_dir = ""
        try:
            current = self._current_input_path()
            start_dir = str(current.parent)
        except Exception:
            start_dir = str(Path(self._normalized_input_text() or ".").expanduser())
        svg_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open SVG with metadata",
            start_dir,
            "SVG (*.svg)",
        )
        if not svg_path:
            return
        try:
            meta = self._read_svg_metadata(Path(svg_path))
            self._load_from_svg_metadata(meta)
            self.lbl_status.setText(f"Restored from SVG metadata:\n{Path(svg_path).name}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Open SVG failed", str(e))

    def _load_from_svg_metadata(self, meta: dict):
        data_path = meta.get("data_path", None)
        img = None
        scale = None
        units = None
        used_data_path = None

        if data_path:
            p = Path(data_path).expanduser()
            if p.exists() and p.is_file():
                if self._is_plain_image_suffix(p):
                    img = self._load_plain_image(p)
                    scale = 0.0
                    units = "nm^-1"
                    used_data_path = p
                else:
                    img, scale, units, _size = load_hyperspy_2d(p)
                    used_data_path = p

        if img is None:
            raise ValueError("SVG metadata does not contain a valid source file path.")

        self.edit_path.setText(str(used_data_path))
        self._apply_loaded_image(
            img=img,
            data_path=used_data_path,
            key=str(used_data_path.name),
            scale=scale,
            units=units,
        )
        self._restore_gui_from_metadata(meta)

    def _restore_gui_from_metadata(self, meta: dict):
        def _safe_float(v, default=None):
            try:
                return float(v)
            except Exception:
                return default

        with QtCore.QSignalBlocker(self.spin_clip):
            cv = _safe_float(meta.get("clip_percentile", None), float(self.spin_clip.value()))
            self.spin_clip.setValue(float(np.clip(cv, 0.1, 50.0)))
        with QtCore.QSignalBlocker(self.spin_rotation):
            rv = _safe_float(meta.get("saed_rotation_deg", None), float(self.spin_rotation.value()))
            self.spin_rotation.setValue(float(np.clip(rv, -180.0, 180.0)))
        with QtCore.QSignalBlocker(self.spin_zoom):
            zv = _safe_float(meta.get("saed_zoom_factor", None), float(self.spin_zoom.value()))
            self.spin_zoom.setValue(float(np.clip(zv, 0.2, 1.0)))
        with QtCore.QSignalBlocker(self.chk_log):
            self.chk_log.setChecked(bool(meta.get("log_intensity", False)))

        cmap_name = str(meta.get("cmap", self.combo_cmap.currentText()))
        idx = self.combo_cmap.findText(cmap_name)
        if idx >= 0:
            with QtCore.QSignalBlocker(self.combo_cmap):
                self.combo_cmap.setCurrentIndex(idx)

        center_method = str(meta.get("center_method", self._center_method or "ring")).lower()
        if center_method not in ("midpoint", "ring", "circle"):
            center_method = "ring"
        self._set_center_method(center_method)

        crystal_mode = str(meta.get("crystal_mode", self._crystal_mode or "poly")).lower()
        if crystal_mode == "single":
            self.btn_single.setChecked(True)
            self._crystal_mode = "single"
        else:
            self.btn_poly.setChecked(True)
            self._crystal_mode = "poly"

        center_xy = meta.get("center_xy", None)
        if isinstance(center_xy, (list, tuple)) and len(center_xy) >= 2:
            cx = _safe_float(center_xy[0], None)
            cy = _safe_float(center_xy[1], None)
            if cx is not None and cy is not None:
                self.center_xy = (cx, cy)
        center_circle = meta.get("center_circle", None)
        if isinstance(center_circle, dict):
            rr = _safe_float(center_circle.get("radius_px", None), None)
            if rr is not None and rr > 0 and self.display_image is not None:
                self._ensure_center_circle_roi()
                if self._center_circle_roi is not None and self.center_xy is not None:
                    self._updating_center_circle = True
                    try:
                        cx = float(self.center_xy[0])
                        cy = float(self.center_xy[1])
                        r = float(rr)
                        self._center_circle_roi.setPos([cx - r, cy - r])
                        self._center_circle_roi.setSize([2.0 * r, 2.0 * r])
                    finally:
                        self._updating_center_circle = False

        self.picks = []
        for p in meta.get("picks", []) or []:
            q = _safe_float((p or {}).get("q", None), None)
            d = _safe_float((p or {}).get("d_ang", None), None)
            if q is None or q <= 0:
                continue
            if d is None:
                d = 10.0 / q
            mode = str((p or {}).get("mode", self._crystal_mode))
            pick = {"mode": mode, "q": q, "d_ang": d}
            kx = _safe_float((p or {}).get("kx", None), None)
            ky = _safe_float((p or {}).get("ky", None), None)
            ang = _safe_float((p or {}).get("angle_deg", None), None)
            if kx is not None:
                pick["kx"] = kx
            if ky is not None:
                pick["ky"] = ky
            if ang is not None:
                pick["angle_deg"] = ang
            self.picks.append(pick)

        self.display_image = self._rotated_image(self.raw_image)
        self.compute_profile()

        if self.profile_r.size > 0:
            qmin = float(np.min(self.profile_r))
            qmax = float(np.max(self.profile_r))
            pr = meta.get("plot_x_range", None)
            if isinstance(pr, (list, tuple)) and len(pr) >= 2:
                x0 = _safe_float(pr[0], qmin)
                x1 = _safe_float(pr[1], qmax)
                self.plot_x_bar.set_values(qmin, qmax, x0, x1)
                self._plot_x_initialized = True
                self._plot_auto_peak_pending = False
            else:
                self._plot_x_initialized = False
                self._plot_auto_peak_pending = True

        self._ensure_bg_regions()
        bg = meta.get("bg_fitting_range", {}) or {}
        if self.profile_r.size > 0:
            qmin = float(np.min(self.profile_r))
            qmax = float(np.max(self.profile_r))
            ls = _safe_float(bg.get("low_start", None), qmin + 0.05 * (qmax - qmin))
            le = _safe_float(bg.get("low_end", None), qmin + 0.18 * (qmax - qmin))
            hs = _safe_float(bg.get("high_start", None), qmin + 0.70 * (qmax - qmin))
            ls = float(np.clip(ls, qmin, qmax))
            le = float(np.clip(le, qmin, qmax))
            hs = float(np.clip(hs, qmin, qmax))
            if le <= ls:
                le = min(qmax, ls + 1e-6)
            if hs <= le:
                hs = min(qmax, le + 1e-6)
            self._updating_bg_region = True
            self._bg_region_low.setRegion((ls, le))
            self._bg_region_high.setRegion((hs, qmax))
            self._updating_bg_region = False
            self._bg_initialized = True

        self.fit_background_and_update()
        self.refresh_profile_plot()
        self.refresh_saed_view()
        self.refresh_center_overlay()
        self.refresh_pick_overlays()
        self.refresh_table()

    def _image_for_saed_display(self, image: np.ndarray) -> np.ndarray:
        arr = np.asarray(image, dtype=np.float32)
        if not self.chk_log.isChecked():
            return arr
        arr = arr - float(np.min(arr))
        return np.log1p(np.clip(arr, 0.0, None))

    def _apply_cmap_to_item(self, image_item: pg.ImageItem):
        cmap_name = self.combo_cmap.currentText() if hasattr(self, "combo_cmap") else "gray"
        try:
            from matplotlib import colormaps
            lut = colormaps.get_cmap(cmap_name)(np.linspace(0.0, 1.0, 256))[:, :3]
            lut = np.clip(lut * 255.0, 0, 255).astype(np.ubyte)
            image_item.setLookupTable(lut)
        except Exception:
            image_item.setLookupTable(None)

    def _extract_square_crop(self, image: np.ndarray, center_xy, zoom: float):
        h, w = image.shape
        cx, cy = float(center_xy[0]), float(center_xy[1])
        half = int(max(2, min(h, w) * 0.4 * zoom))
        half = int(min(half, h // 2, w // 2))
        if half < 2:
            return image.copy()
        cx_i = int(np.clip(round(cx), half, w - half))
        cy_i = int(np.clip(round(cy), half, h - half))
        x0, x1 = cx_i - half, cx_i + half
        y0, y1 = cy_i - half, cy_i + half
        return image[y0:y1, x0:x1]

    @staticmethod
    def _ticks_125(vmin: float, vmax: float, target_n: int = 6):
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            return []
        span = float(vmax - vmin)
        raw = span / max(2, int(target_n))
        if raw <= 0:
            return []
        expn = math.floor(math.log10(raw))
        base = raw / (10.0 ** expn)
        if base <= 1.0:
            step_base = 1.0
        elif base <= 2.0:
            step_base = 2.0
        elif base <= 5.0:
            step_base = 5.0
        else:
            step_base = 10.0
        step = step_base * (10.0 ** expn)
        t0 = math.ceil(vmin / step) * step
        ticks = []
        t = t0
        guard = 0
        while t <= vmax + 1e-12 and guard < 200:
            ticks.append(round(float(t), 10))
            t += step
            guard += 1
        return ticks

    def _apply_saed_ticks_gui(self, rect: QtCore.QRectF):
        x_ticks = self._ticks_125(float(rect.left()), float(rect.right()), target_n=6)
        y_ticks = self._ticks_125(float(rect.top()), float(rect.bottom()), target_n=6)
        if x_ticks:
            self.saed_plot.getAxis("bottom").setTicks([[(t, f"{t:g}") for t in x_ticks]])
        if y_ticks:
            self.saed_plot.getAxis("left").setTicks([[(t, f"{t:g}") for t in y_ticks]])

    def _rotated_image(self, img: np.ndarray) -> np.ndarray:
        angle = float(self.spin_rotation.value()) if hasattr(self, "spin_rotation") else 0.0
        if abs(angle) < 1e-6:
            return img
        try:
            from scipy.ndimage import rotate
            return rotate(img, angle=angle, reshape=False, order=1, mode="nearest")
        except Exception:
            return img

    def on_view_option_changed(self):
        if self.raw_image is None:
            return
        self.display_image = self._rotated_image(self.raw_image)
        self.refresh_all()

    def refresh_all(self):
        self.refresh_raw_view()
        self.compute_profile()
        if self.profile_r.size > 0 and ((not self._plot_x_initialized) or self._plot_auto_peak_pending):
            qmin = float(np.min(self.profile_r))
            qmax = float(np.max(self.profile_r))
            y_ref = self.profile_sub if self.profile_sub.size == self.profile_r.size else self.profile_i_raw
            x_first_peak = self._find_first_peak_x(self.profile_r, y_ref)
            x_first_peak = float(np.clip(x_first_peak, qmin, qmax))
            self.plot_x_bar.set_values(qmin, qmax, x_first_peak, qmax)
            self._plot_x_initialized = True
            self._plot_auto_peak_pending = False
        if not self._bg_initialized and self.profile_r.size > 0:
            self._set_default_bg_regions()
            self._bg_initialized = True
        self.fit_background_and_update()
        self.refresh_profile_plot()
        self.refresh_saed_view()
        self.refresh_center_overlay()
        self.refresh_pick_overlays()
        self.refresh_table()
        self._update_center_display()

    def refresh_raw_view(self):
        if self.display_image is None:
            self.raw_img_item.clear()
            return
        clip = float(self.spin_clip.value())
        img8 = clipped_8bit_image(self.display_image, clip).astype(np.float32)
        self._apply_cmap_to_item(self.raw_img_item)
        self.raw_img_item.setImage(img8, autoLevels=False)
        self.raw_img_item.setLevels((0, 255))

    def refresh_center_overlay(self):
        pts = [{"pos": p, "data": 1} for p in self.center_pick_points]
        self.raw_scatter.setData(pts)
        self._ensure_center_circle_roi()
        show_circle = (self._center_method == "circle") and (self.display_image is not None)
        if self._center_circle_roi is not None:
            self._center_circle_roi.setVisible(show_circle)
            if show_circle and self.center_xy is not None and (not self._updating_center_circle):
                self._updating_center_circle = True
                try:
                    r = float(self._center_circle_roi.size().x()) * 0.5
                    if not np.isfinite(r) or r <= 1.0:
                        h, w = self.display_image.shape[:2]
                        r = float(max(12.0, 0.2 * min(h, w)))
                    cx, cy = float(self.center_xy[0]), float(self.center_xy[1])
                    self._center_circle_roi.setPos([cx - r, cy - r])
                    self._center_circle_roi.setSize([2.0 * r, 2.0 * r])
                finally:
                    self._updating_center_circle = False
        self._update_center_marker_live()
        self._update_center_display()

    def _update_center_marker_live(self):
        if self.center_xy is None:
            if self._center_marker is not None:
                try:
                    self.raw_plot.removeItem(self._center_marker)
                except Exception:
                    pass
                self._center_marker = None
            return
        if self._center_marker is None:
            self._center_marker = pg.ScatterPlotItem(
                size=14,
                symbol="+",
                pen=pg.mkPen(80, 240, 255, width=2),
                brush=pg.mkBrush(80, 240, 255, 80),
            )
            self.raw_plot.addItem(self._center_marker)
        self._center_marker.setData([{"pos": (float(self.center_xy[0]), float(self.center_xy[1]))}])

    def _update_center_display(self):
        if not hasattr(self, "edit_center_xy"):
            return
        if self.center_xy is None:
            self.edit_center_xy.setText("x=--, y=--")
            return
        cx = float(self.center_xy[0])
        cy = float(self.center_xy[1])
        self.edit_center_xy.setText(f"x={cx:.2f}, y={cy:.2f}")

    def _ensure_center_circle_roi(self):
        if self._center_circle_roi is not None:
            return
        if self.display_image is None:
            return
        h, w = self.display_image.shape[:2]
        if self.center_xy is None:
            cx, cy = 0.5 * w, 0.5 * h
        else:
            cx, cy = float(self.center_xy[0]), float(self.center_xy[1])
        r = float(max(12.0, 0.2 * min(h, w)))
        self._center_circle_roi = pg.CircleROI(
            [cx - r, cy - r],
            [2.0 * r, 2.0 * r],
            movable=True,
            resizable=True,
            pen=pg.mkPen(80, 240, 255, width=1.8),
        )
        self.raw_plot.addItem(self._center_circle_roi)
        self._center_circle_roi.sigRegionChanged.connect(self._on_center_circle_changed)
        self._center_circle_roi.sigRegionChangeFinished.connect(self._on_center_circle_change_finished)
        self._center_circle_roi.setZValue(20)
        self._center_circle_roi.setVisible(False)

    def _on_center_circle_changed(self):
        if self._center_circle_roi is None or self._updating_center_circle:
            return
        if self.display_image is None:
            return
        # User started moving again: cancel any pending heavy recompute.
        if self._center_update_timer.isActive():
            self._center_update_timer.stop()
        self._updating_center_circle = True
        try:
            h, w = self.display_image.shape[:2]
            pos = self._center_circle_roi.pos()
            size = self._center_circle_roi.size()
            d = float(max(2.0, min(float(size.x()), float(size.y()))))
            r = 0.5 * d
            cx = float(pos.x()) + r
            cy = float(pos.y()) + r
            cx = float(np.clip(cx, 0.0, float(w - 1)))
            cy = float(np.clip(cy, 0.0, float(h - 1)))
            max_r = max(2.0, min(cx, cy, float(w - 1) - cx, float(h - 1) - cy))
            r = float(np.clip(r, 2.0, max_r))
            self._center_circle_roi.setPos([cx - r, cy - r])
            self._center_circle_roi.setSize([2.0 * r, 2.0 * r])
            self.center_xy = (cx, cy)
            self._active_zone = "center"
            # Lightweight feedback during drag/resize: show live center marker.
            self._update_center_marker_live()
            self._update_center_display()
        finally:
            self._updating_center_circle = False

    def _on_center_circle_change_finished(self):
        if self._center_method != "circle":
            return
        self._schedule_deferred_center_update()

    def _schedule_deferred_center_update(self, delay_ms: int = 480):
        if self._center_update_timer.isActive():
            self._center_update_timer.stop()
        self._center_update_timer.start(int(max(0, delay_ms)))

    def _apply_deferred_center_update(self):
        if self.display_image is None or self.center_xy is None:
            return
        self._plot_auto_peak_pending = True
        self.compute_profile()
        self.fit_background_and_update()
        self.refresh_profile_plot()
        self.refresh_center_overlay()
        self.refresh_pick_overlays()
        self.refresh_saed_view()
        self.refresh_table()

    def clear_center_points(self):
        self.center_pick_points.clear()
        self.refresh_center_overlay()

    def on_crystal_mode_changed(self):
        if self.btn_single.isChecked():
            self._crystal_mode = "single"
            self._set_center_method("midpoint")
            self.lbl_status.setText("Single crystal mode: pick diffraction spots on Zoom SAED.")
        else:
            self._crystal_mode = "poly"
            self._set_center_method("ring")
            self.lbl_status.setText("Poly crystal mode: pick rings.")
        self.clear_picks()

    def _set_center_method(self, method: str):
        m = str(method).strip().lower()
        if m.startswith("r"):
            self._center_method = "ring"
        elif m.startswith("c"):
            self._center_method = "circle"
        else:
            self._center_method = "midpoint"
        if self._center_method == "ring":
            self.btn_method_switch.setText("Ring(3pt)")
        elif self._center_method == "circle":
            self.btn_method_switch.setText("Circle(ROI)")
        else:
            self.btn_method_switch.setText("Midpoint(2pt)")

    def on_method_switch(self):
        if self._center_method == "midpoint":
            self._set_center_method("ring")
        elif self._center_method == "ring":
            self._set_center_method("circle")
        else:
            self._set_center_method("midpoint")
        self.center_pick_points.clear()
        self._plot_auto_peak_pending = True
        self.refresh_all()

    def auto_center(self):
        if self.display_image is None:
            return
        try:
            from skued import autocenter
            mask = np.ones_like(self.display_image, dtype=bool)
            yc, xc = autocenter(self.display_image, mask)
            self.center_xy = (float(xc), float(yc))
            self.center_pick_points.clear()
            self._plot_auto_peak_pending = True
            self.refresh_all()
            self.lbl_status.setText(f"Auto center: ({self.center_xy[0]:.2f}, {self.center_xy[1]:.2f})")
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Auto center failed", str(e))

    def _local_peak(self, x: float, y: float, half: int = 8):
        if self.display_image is None:
            return x, y
        h, w = self.display_image.shape
        xi = int(np.clip(round(x), 0, w - 1))
        yi = int(np.clip(round(y), 0, h - 1))
        xs = max(0, xi - half)
        xe = min(w, xi + half + 1)
        ys = max(0, yi - half)
        ye = min(h, yi + half + 1)
        patch = self.display_image[ys:ye, xs:xe]
        if patch.size == 0:
            return float(xi), float(yi)
        ly, lx = np.unravel_index(np.argmax(patch), patch.shape)
        return float(xs + lx), float(ys + ly)

    def on_raw_clicked(self, ev):
        if self.display_image is None:
            return
        if ev.button() != QtCore.Qt.LeftButton:
            return
        pos = ev.scenePos()
        vb = self.raw_plot.vb
        # Only clicks in the ViewBox data area are valid (exclude axes/ticks/margins).
        if not vb.sceneBoundingRect().contains(pos):
            return
        if self._click_hits_viewbox_overlay(vb, pos):
            return
        p = vb.mapSceneToView(pos)
        # Require click in raw image bounds.
        h, w = self.display_image.shape
        x_raw = float(p.x())
        y_raw = float(p.y())
        if not (0.0 <= x_raw < float(w) and 0.0 <= y_raw < float(h)):
            return
        x, y = self._local_peak(p.x(), p.y())
        self._active_zone = "center"
        mode = self._center_method
        if mode == "circle":
            self.center_xy = (x, y)
        elif mode == "midpoint":
            self.center_pick_points.append((x, y))
            self.center_pick_points = self.center_pick_points[-2:]
            if len(self.center_pick_points) == 2:
                p1, p2 = self.center_pick_points
                self.center_xy = ((p1[0] + p2[0]) * 0.5, (p1[1] + p2[1]) * 0.5)
        else:
            self.center_pick_points.append((x, y))
            self.center_pick_points = self.center_pick_points[-3:]
            if len(self.center_pick_points) == 3:
                try:
                    self.center_xy = circle_center_from_3pts(
                        self.center_pick_points[0],
                        self.center_pick_points[1],
                        self.center_pick_points[2],
                    )
                except Exception as e:
                    self.lbl_status.setText(f"Ring center failed: {e}")
        self._plot_auto_peak_pending = True
        self.refresh_all()

    def _recompute_center_from_points(self):
        mode = self._center_method
        if mode == "midpoint" and len(self.center_pick_points) >= 2:
            p1, p2 = self.center_pick_points[-2], self.center_pick_points[-1]
            self.center_xy = ((p1[0] + p2[0]) * 0.5, (p1[1] + p2[1]) * 0.5)
        elif mode == "ring" and len(self.center_pick_points) >= 3:
            try:
                self.center_xy = circle_center_from_3pts(
                    self.center_pick_points[-3],
                    self.center_pick_points[-2],
                    self.center_pick_points[-1],
                )
            except Exception:
                pass

    def remove_last_center_point(self):
        if not self.center_pick_points:
            return
        self.center_pick_points.pop()
        self._recompute_center_from_points()
        self._plot_auto_peak_pending = True
        self.refresh_all()

    def undo_contextual(self):
        if self._active_zone == "center" and self.center_pick_points:
            self.remove_last_center_point()
            return
        self.remove_last_pick()

    def _center_beam_mask(self, image, center, sigma=30, threshold=0.1):
        h, w = image.shape
        xx, yy = np.meshgrid(np.arange(w), np.arange(h))
        rr = np.sqrt((xx - center[0]) ** 2 + (yy - center[1]) ** 2)
        m = np.exp(-(rr ** 2) / (2.0 * sigma * sigma))
        return m <= threshold

    def compute_profile(self):
        if self.display_image is None or self.center_xy is None:
            self.profile_r = np.zeros(0, dtype=np.float32)
            self.profile_i_raw = np.zeros(0, dtype=np.float32)
            return
        img = self.display_image
        center = self.center_xy
        mask = self._center_beam_mask(img, center=center, threshold=0.1)
        try:
            from skued import azimuthal_average
            r_px, intensity = azimuthal_average(img, center, mask=mask)
            self._using_skued = True
        except Exception:
            self._using_skued = False
            yy, xx = np.indices(img.shape, dtype=np.float32)
            rr = np.hypot(xx - float(center[0]), yy - float(center[1]))
            rr_i = rr.astype(np.int32)
            vals = img[mask]
            bins = rr_i[mask]
            sums = np.bincount(bins.ravel(), weights=vals.ravel())
            cnts = np.bincount(bins.ravel())
            nz = cnts > 0
            profile = np.zeros_like(sums, dtype=np.float32)
            profile[nz] = sums[nz] / cnts[nz]
            r_px = np.arange(profile.size, dtype=np.float32)
            intensity = profile
        r_px = np.asarray(r_px, dtype=np.float32)
        intensity = np.asarray(intensity, dtype=np.float32)
        scale = self.scale_nm_inv_per_px if self.scale_nm_inv_per_px > 0 else 1.0
        self.profile_r = r_px * scale
        self.profile_i_raw = intensity

    def _find_first_peak_x(self, x: np.ndarray, y: np.ndarray) -> float:
        if x.size < 3 or y.size < 3:
            return float(x[0]) if x.size else 0.0
        xx = np.asarray(x, dtype=np.float64)
        yy = np.asarray(y, dtype=np.float64)
        valid = np.isfinite(xx) & np.isfinite(yy)
        xx = xx[valid]
        yy = yy[valid]
        if xx.size < 3:
            return float(x[0]) if x.size else 0.0

        x_min = float(np.min(xx))
        x_max = float(np.max(xx))
        x_thr = x_min + 0.02 * max(x_max - x_min, 1e-12)

        try:
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(yy)
            for pi in peaks:
                xp = float(xx[pi])
                if xp >= x_thr:
                    return xp
        except Exception:
            pass

        for i in range(1, yy.size - 1):
            if yy[i] > yy[i - 1] and yy[i] >= yy[i + 1] and xx[i] >= x_thr:
                return float(xx[i])
        return x_min

    def _set_default_bg_regions(self):
        if self.profile_r.size == 0:
            return
        qmin = float(np.min(self.profile_r))
        qmax = float(np.max(self.profile_r))
        rng = max(qmax - qmin, 1e-9)
        low_start = qmin + 0.05 * rng
        low_end = qmin + 0.18 * rng
        high_start = qmin + 0.7 * rng
        self._updating_bg_region = True
        self._ensure_bg_regions()
        self._bg_region_low.setRegion((low_start, low_end))
        self._bg_region_high.setRegion((high_start, qmax))
        self._updating_bg_region = False

    def _ensure_bg_regions(self):
        if self._bg_region_low is not None and self._bg_region_high is not None:
            return
        self._bg_region_low = pg.LinearRegionItem(values=[0.0, 0.1], movable=True, brush=(60, 180, 255, 35))
        self._bg_region_high = pg.LinearRegionItem(values=[0.8, 1.0], movable=True, brush=(60, 180, 255, 25))
        self.bg_plot.addItem(self._bg_region_low)
        self.bg_plot.addItem(self._bg_region_high)
        self._bg_region_items = [self._bg_region_low, self._bg_region_high]
        self._bg_region_low.sigRegionChanged.connect(self.on_bg_region_changed)
        self._bg_region_high.sigRegionChanged.connect(self.on_bg_region_changed)

    def on_bg_region_changed(self):
        if self._updating_bg_region:
            return
        self.fit_background_and_update()
        self.refresh_profile_plot()

    def on_plot_range_changed(self, x_min=None, x_max=None):
        if self.profile_r.size == 0:
            self.lbl_plot_x.setText("0.00 ~ 0.00")
            return
        if x_min is None or x_max is None:
            x_min, x_max = self.plot_x_bar.values()
        qmin = float(np.min(self.profile_r))
        qmax = float(np.max(self.profile_r))
        x_min = float(np.clip(x_min, qmin, qmax))
        x_max = float(np.clip(x_max, qmin, qmax))
        if x_max <= x_min:
            x_max = min(qmax, x_min + max((qmax - qmin) * 1e-3, 1e-9))
        self.lbl_plot_x.setText(f"{x_min:.2f} ~ {x_max:.2f}")
        self.refresh_profile_plot()

    def fit_background_and_update(self):
        if self.profile_r.size < 10 or self.profile_i_raw.size < 10:
            self.profile_bg = np.zeros_like(self.profile_r)
            self.profile_sub = self.profile_i_raw.copy()
            return
        self._ensure_bg_regions()
        ls, le = self._bg_region_low.getRegion()
        hs, _h1 = self._bg_region_high.getRegion()

        qmin = float(np.min(self.profile_r))
        qmax = float(np.max(self.profile_r))
        ls = float(np.clip(ls, qmin, qmax))
        le = float(np.clip(le, qmin, qmax))
        hs = float(np.clip(hs, qmin, qmax))
        if le <= ls:
            le = min(qmax, ls + 1e-6)
        if hs <= le:
            hs = min(qmax, le + 1e-6)

        self._updating_bg_region = True
        self._bg_region_low.setRegion((ls, le))
        self._bg_region_high.setRegion((hs, qmax))
        self._updating_bg_region = False

        mask = ((self.profile_r >= ls) & (self.profile_r <= le)) | (self.profile_r >= hs)
        x_fit = self.profile_r[mask]
        y_fit = self.profile_i_raw[mask]

        bg = np.zeros_like(self.profile_i_raw)
        ok = False
        try:
            from scipy.optimize import curve_fit
            p0 = (float(np.median(y_fit)), -1.0, float(np.min(y_fit)))
            popt, _ = curve_fit(_power_law, x_fit, y_fit, p0=p0, maxfev=30000)
            bg = _power_law(self.profile_r, *popt)
            ok = True
        except Exception:
            pass

        if not ok:
            bg[:] = float(np.median(self.profile_i_raw))
        self.profile_bg = np.asarray(bg, dtype=np.float32)
        self.profile_sub = np.asarray(self.profile_i_raw - self.profile_bg, dtype=np.float32)
        self.lbl_bg.setText(f"low_start={ls:.3f}  low_end={le:.3f}  high_start={hs:.3f}")
        self.bg_curve_raw.setData(self.profile_r, self.profile_i_raw)
        self.bg_curve_fit.setData(self.profile_r, self.profile_bg)
        self.bg_curve_sub.setData(self.profile_r, self.profile_sub)
        if not self._bg_ylim_initialized:
            finite = np.isfinite(self.profile_i_raw)
            if np.any(finite):
                y_max = float(np.max(self.profile_i_raw[finite]))
                y_max = max(y_max, 1.0)
                self.bg_plot.setYRange(0.0, y_max * 1.05, padding=0.0)
                self._bg_ylim_initialized = True

    def refresh_profile_plot(self):
        if self.profile_r.size == 0:
            self.profile_curve.setData([], [])
            return
        y = self.profile_sub if self.profile_sub.size == self.profile_r.size else self.profile_i_raw
        if self.chk_log.isChecked():
            y_plot = np.log10(np.clip(y - np.min(y) + 1.0, 1e-9, None))
            self.profile_plot.setLabel("left", "Intensity (log10)")
        else:
            y_plot = y
            self.profile_plot.setLabel("left", "Intensity")
        self.profile_curve.setData(self.profile_r, y_plot)
        qmin = float(np.min(self.profile_r))
        qmax = float(np.max(self.profile_r))
        x_min, x_max = self.plot_x_bar.values() if self._plot_x_initialized else (qmin, qmax)
        x_min = float(np.clip(x_min, qmin, qmax))
        x_max = float(np.clip(x_max, qmin, qmax))
        if x_max <= x_min:
            x_max = min(qmax, x_min + max((qmax - qmin) * 1e-3, 1e-9))
        vb = self.profile_plot.getPlotItem().vb
        mask = (self.profile_r >= x_min) & (self.profile_r <= x_max) & np.isfinite(y_plot)
        if not np.any(mask):
            mask = np.isfinite(y_plot)
        if np.any(mask):
            ys = y_plot[mask]
            y0, y1 = float(np.min(ys)), float(np.max(ys))
            if y1 <= y0:
                c = 0.5 * (y0 + y1)
                y0, y1 = c - 0.5, c + 0.5
            pad = max(0.015 * (y1 - y0), 1e-9)
            vb.setRange(
                xRange=(x_min, x_max),
                yRange=(y0 - pad, y1 + pad),
                padding=0.0,
                disableAutoRange=True,
            )
        else:
            vb.setRange(xRange=(x_min, x_max), padding=0.0, disableAutoRange=True)
        self.lbl_plot_x.setText(f"{x_min:.2f} ~ {x_max:.2f}")

    def refresh_saed_view(self):
        if self.display_image is None:
            self.saed_img_item.clear()
            return
        if self.center_xy is None:
            self.center_xy = (self.display_image.shape[1] * 0.5, self.display_image.shape[0] * 0.5)
        zoom = float(self.spin_zoom.value())
        src = self._image_for_saed_display(self.display_image)
        crop = self._extract_square_crop(src, self.center_xy, zoom)
        self._saed_crop = np.asarray(crop, dtype=np.float32)
        clip = float(self.spin_clip.value())
        crop8 = clipped_8bit_image(crop, clip).astype(np.float32)
        self._apply_cmap_to_item(self.saed_img_item)
        self.saed_img_item.setImage(crop8, autoLevels=False)
        self.saed_img_item.setLevels((0, 255))

        k_scale = self.scale_nm_inv_per_px if self.scale_nm_inv_per_px > 0 else 1.0
        cx_k = (crop.shape[1] * 0.5) * k_scale
        cy_k = (crop.shape[0] * 0.5) * k_scale
        rect = QtCore.QRectF(-cx_k, -cy_k, crop.shape[1] * k_scale, crop.shape[0] * k_scale)
        self._saed_display_rect = rect
        self.saed_img_item.setRect(rect)
        self.saed_plot.setXRange(rect.left(), rect.right(), padding=0.01)
        self.saed_plot.setYRange(rect.top(), rect.bottom(), padding=0.01)
        self.saed_plot.setAspectLocked(True, ratio=1.0)
        self._apply_saed_ticks_gui(rect)
        self.saed_cross_v.setValue(0.0)
        self.saed_cross_h.setValue(0.0)
        self.saed_cross_v.setVisible(True)
        self.saed_cross_h.setVisible(True)
        self.refresh_pick_overlays()

    def on_profile_clicked(self, ev):
        if self._crystal_mode == "single":
            return
        if self.profile_r.size == 0:
            return
        if ev.button() != QtCore.Qt.LeftButton:
            return
        pos = ev.scenePos()
        if not self.profile_plot.sceneBoundingRect().contains(pos):
            return
        vb = self.profile_plot.getPlotItem().vb
        p = vb.mapSceneToView(pos)
        q = float(p.x())
        self._active_zone = "pick"
        qmin = float(np.min(self.profile_r))
        qmax = float(np.max(self.profile_r))
        q = float(np.clip(q, qmin, qmax))
        if q <= 0:
            return
        self._add_pick(q)

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

    def on_saed_clicked(self, ev):
        if self.profile_r.size == 0:
            return
        if ev.button() != QtCore.Qt.LeftButton:
            return
        pos = ev.scenePos()
        vb = self.saed_plot.vb
        # Only clicks in the ViewBox data area are valid (exclude axes/ticks/margins).
        if not vb.sceneBoundingRect().contains(pos):
            return
        if self._click_hits_viewbox_overlay(vb, pos):
            return
        p = vb.mapSceneToView(pos)
        # Also require the mapped data point to be inside the SAED image rect.
        rect = self._saed_display_rect
        if rect is None or (not rect.contains(QtCore.QPointF(float(p.x()), float(p.y())))):
            return
        self._active_zone = "pick"
        if self._crystal_mode == "single":
            peak = self._fit_saed_peak_from_k(float(p.x()), float(p.y()))
            if peak is None:
                return
            self._add_single_pick(peak["kx"], peak["ky"])
        else:
            q = float(np.hypot(p.x(), p.y()))
            qmin = float(np.min(self.profile_r))
            qmax = float(np.max(self.profile_r))
            q = float(np.clip(q, qmin, qmax))
            if q <= 0:
                return
            self._add_pick(q)

    def _add_pick(self, q: float):
        d_ang = 10.0 / q
        self.picks.append({"mode": "poly", "q": float(q), "d_ang": float(d_ang), "angle_deg": float("nan")})
        self.refresh_pick_overlays()
        self.refresh_table()

    def _add_single_pick(self, kx: float, ky: float):
        q = float(np.hypot(kx, ky))
        if q <= 1e-12:
            return
        d_ang = 10.0 / q
        ang = 0.0 if len(self.picks) == 0 else self._angle_to_first_pick_deg(kx, ky)
        self.picks.append(
            {
                "mode": "single",
                "kx": float(kx),
                "ky": float(ky),
                "q": float(q),
                "d_ang": float(d_ang),
                "angle_deg": float(ang),
            }
        )
        self.refresh_pick_overlays()
        self.refresh_table()

    def _angle_to_first_pick_deg(self, kx: float, ky: float) -> float:
        if not self.picks:
            return float("nan")
        first = self.picks[0]
        kx0 = float(first.get("kx", 0.0))
        ky0 = float(first.get("ky", 0.0))
        n0 = float(np.hypot(kx0, ky0))
        n1 = float(np.hypot(kx, ky))
        if n0 <= 1e-12 or n1 <= 1e-12:
            return float("nan")
        cosang = float(np.clip((kx0 * kx + ky0 * ky) / (n0 * n1), -1.0, 1.0))
        return float(np.degrees(np.arccos(cosang)))

    def _fit_saed_peak_from_k(self, kx_guess: float, ky_guess: float):
        if self._saed_crop is None or self._saed_display_rect is None:
            return None
        img = np.asarray(self._saed_crop, dtype=np.float32)
        rect = self._saed_display_rect
        h, w = img.shape
        if h < 5 or w < 5:
            return None

        # k-space -> crop pixel
        kx0 = float(rect.left())
        ky0 = float(rect.top())
        dkx = float(rect.width()) / float(w)
        dky = float(rect.height()) / float(h)
        if dkx == 0.0 or dky == 0.0:
            return None
        xg = (float(kx_guess) - kx0) / dkx
        yg = (float(ky_guess) - ky0) / dky
        xi = int(np.clip(round(xg), 0, w - 1))
        yi = int(np.clip(round(yg), 0, h - 1))

        half = 8
        xs = max(0, xi - half)
        xe = min(w, xi + half + 1)
        ys = max(0, yi - half)
        ye = min(h, yi + half + 1)
        patch = img[ys:ye, xs:xe]
        if patch.size == 0:
            return None

        # robust local fit (fallback to local max)
        yloc, xloc = np.unravel_index(np.argmax(patch), patch.shape)
        x_peak = float(xs + xloc)
        y_peak = float(ys + yloc)
        try:
            from scipy.optimize import curve_fit

            ny, nx = patch.shape
            X, Y = np.meshgrid(np.arange(nx, dtype=np.float32), np.arange(ny, dtype=np.float32))

            def g2d(xy, amp, x0, y0, sx, sy, off):
                x, y = xy
                return (amp * np.exp(-(((x - x0) ** 2) / (2 * sx ** 2) + ((y - y0) ** 2) / (2 * sy ** 2))) + off).ravel()

            p0 = (float(np.max(patch) - np.min(patch)), float(nx) * 0.5, float(ny) * 0.5, 2.5, 2.5, float(np.min(patch)))
            popt, _ = curve_fit(g2d, (X, Y), patch.ravel(), p0=p0, maxfev=10000)
            xf = float(popt[1])
            yf = float(popt[2])
            if 0 <= xf < nx and 0 <= yf < ny:
                x_peak = float(xs + xf)
                y_peak = float(ys + yf)
        except Exception:
            pass

        kx = kx0 + (x_peak + 0.5) * dkx
        ky = ky0 + (y_peak + 0.5) * dky
        return {"kx": float(kx), "ky": float(ky)}

    def _clear_overlay_items(self):
        for it in self._arc_items:
            try:
                self.saed_plot.removeItem(it)
            except Exception:
                pass
        for it in self._spot_items:
            try:
                self.saed_plot.removeItem(it)
            except Exception:
                pass
        for it in self._spot_label_items:
            try:
                self.saed_plot.removeItem(it)
            except Exception:
                pass
        for it in self._vline_items:
            try:
                self.profile_plot.removeItem(it)
            except Exception:
                pass
        self._arc_items.clear()
        self._spot_items.clear()
        self._spot_label_items.clear()
        self._vline_items.clear()

    def refresh_pick_overlays(self):
        self._clear_overlay_items()
        n = max(1, len(self.picks))
        for i, item in enumerate(self.picks):
            c = self._pick_qcolor(i, n)
            pen = pg.mkPen(c, width=1.8)
            q = float(item.get("q", 0.0))
            vline = pg.InfiniteLine(angle=90, pos=q, movable=False, pen=pen)
            self.profile_plot.addItem(vline)
            self._vline_items.append(vline)

            if item.get("mode", self._crystal_mode) == "single":
                kx = float(item.get("kx", 0.0))
                ky = float(item.get("ky", 0.0))
                spot = pg.ScatterPlotItem(
                    [kx], [ky], size=8, pen=pg.mkPen(c, width=1.5), brush=pg.mkBrush(c)
                )
                self.saed_plot.addItem(spot)
                self._spot_items.append(spot)
                lab = pg.TextItem(text=str(i + 1), anchor=(0, 1), color=c)
                lab.setPos(kx, ky)
                self.saed_plot.addItem(lab)
                self._spot_label_items.append(lab)
            else:
                r = float(item["q"])
                arc_xy = self._quarter_arc_xy(r)
                if arc_xy is None:
                    continue
                x_arc, y_arc = arc_xy
                arc_item = pg.PlotDataItem(x_arc, y_arc, pen=pen)
                self.saed_plot.addItem(arc_item)
                self._arc_items.append(arc_item)

    def _pick_qcolor(self, i: int, n: int):
        color_hex = self._pick_color_hex[i % len(self._pick_color_hex)]
        return QtGui.QColor(color_hex)

    def _quarter_arc_xy(self, radius: float, npts: int = 160):
        if radius <= 0:
            return None
        rect = self._saed_display_rect
        if rect is None or rect.isNull():
            return None

        # Draw in the first quadrant (0°~90°) and truncate to visible SAED image rect.
        x_max = float(max(0.0, rect.right()))
        y_max = float(max(0.0, rect.bottom()))
        if x_max <= 0.0 or y_max <= 0.0:
            return None

        start_deg = 0.0
        end_deg = 90.0
        if x_max < radius:
            start_deg = max(start_deg, math.degrees(math.acos(np.clip(x_max / radius, -1.0, 1.0))))
        if y_max < radius:
            end_deg = min(end_deg, math.degrees(math.asin(np.clip(y_max / radius, -1.0, 1.0))))
        if end_deg <= start_deg:
            return None

        tt = np.linspace(np.deg2rad(start_deg), np.deg2rad(end_deg), int(max(16, npts)))
        xx = radius * np.cos(tt)
        yy = radius * np.sin(tt)
        m = (
            (xx >= float(rect.left())) & (xx <= float(rect.right())) &
            (yy >= float(rect.top())) & (yy <= float(rect.bottom()))
        )
        if not np.any(m):
            return None
        return xx[m], yy[m]

    def refresh_table(self):
        self.table.setRowCount(len(self.picks))
        n = max(1, len(self.picks))
        # Recompute single-crystal angles relative to the first picked spot.
        if self._crystal_mode == "single" and self.picks:
            kx0 = float(self.picks[0].get("kx", 0.0))
            ky0 = float(self.picks[0].get("ky", 0.0))
            n0 = float(np.hypot(kx0, ky0))
            for i, p in enumerate(self.picks):
                if i == 0:
                    p["angle_deg"] = 0.0
                    continue
                kx = float(p.get("kx", 0.0))
                ky = float(p.get("ky", 0.0))
                n1 = float(np.hypot(kx, ky))
                if n0 <= 1e-12 or n1 <= 1e-12:
                    p["angle_deg"] = float("nan")
                else:
                    cosang = float(np.clip((kx0 * kx + ky0 * ky) / (n0 * n1), -1.0, 1.0))
                    p["angle_deg"] = float(np.degrees(np.arccos(cosang)))

        for i, item in enumerate(self.picks, start=1):
            color_item = QtWidgets.QTableWidgetItem("━")
            color_item.setTextAlignment(QtCore.Qt.AlignCenter)
            qcolor = self._pick_qcolor(i - 1, n)
            color_item.setForeground(QtGui.QBrush(qcolor))
            font = color_item.font()
            font.setBold(True)
            color_item.setFont(font)
            self.table.setItem(i - 1, 0, color_item)

            ang = item.get("angle_deg", float("nan"))
            ang_str = f"{float(ang):.2f}" if np.isfinite(float(ang)) else "—"
            vals = [str(i), f"{item['q']:.4f}", f"{item['d_ang']:.4f}", ang_str]
            for j, txt in enumerate(vals):
                cell = QtWidgets.QTableWidgetItem(txt)
                cell.setTextAlignment(QtCore.Qt.AlignCenter)
                self.table.setItem(i - 1, j + 1, cell)

    def clear_picks(self):
        self.picks.clear()
        self.refresh_pick_overlays()
        self.refresh_table()

    def remove_last_pick(self):
        if not self.picks:
            return
        self.picks.pop()
        self.refresh_pick_overlays()
        self.refresh_table()

    def _metadata_dict(self):
        if self._bg_region_low is not None and self._bg_region_high is not None:
            ls, le = self._bg_region_low.getRegion()
            hs, _ = self._bg_region_high.getRegion()
        else:
            ls = le = hs = None
        center_circle = None
        if self._center_circle_roi is not None:
            try:
                size = self._center_circle_roi.size()
                center_circle = {
                    "radius_px": float(max(float(size.x()), float(size.y())) * 0.5),
                }
            except Exception:
                center_circle = None
        return {
            "root_path": self.root_path,
            "key": self.key,
            "data_path": self.image_path.as_posix() if self.image_path is not None else None,
            "center_xy": [float(self.center_xy[0]), float(self.center_xy[1])] if self.center_xy is not None else None,
            "scale_nm_inv_per_px": float(self.scale_nm_inv_per_px),
            "scale_units": self.scale_units,
            "intensity_profile_raw": {
                "q_nm_inv": self.profile_r.tolist() if self.profile_r.size else [],
                "intensity_raw": self.profile_i_raw.tolist() if self.profile_i_raw.size else [],
                "intensity_bg": self.profile_bg.tolist() if self.profile_bg.size else [],
                "intensity_subtracted": self.profile_sub.tolist() if self.profile_sub.size else [],
            },
            "bg_fitting_range": {"low_start": ls, "low_end": le, "high_start": hs},
            "clip_percentile": float(self.spin_clip.value()),
            "log_intensity": bool(self.chk_log.isChecked()),
            "cmap": self.combo_cmap.currentText() if hasattr(self, "combo_cmap") else "gray",
            "saed_rotation_deg": float(self.spin_rotation.value()),
            "saed_zoom_factor": float(self.spin_zoom.value()),
            "center_method": self._center_method,
            "center_circle": center_circle,
            "crystal_mode": self._crystal_mode,
            "plot_x_range": list(self.plot_x_bar.values()) if self._plot_x_initialized else None,
            "picks": [
                {
                    "mode": str(p.get("mode", self._crystal_mode)),
                    "q": float(p["q"]),
                    "d_ang": float(p["d_ang"]),
                    "kx": (float(p["kx"]) if "kx" in p else None),
                    "ky": (float(p["ky"]) if "ky" in p else None),
                    "angle_deg": (float(p["angle_deg"]) if "angle_deg" in p and np.isfinite(float(p["angle_deg"])) else None),
                }
                for p in self.picks
            ],
            "azimuthal_average_backend": "skued" if self._using_skued else "numpy_fallback",
        }

    def save_svg_with_table(self, dpi: int = 300):
        if self.display_image is None:
            QtWidgets.QMessageBox.warning(self, "Save", "No image loaded.")
            return
        dpi = int(dpi)
        if dpi <= 0:
            dpi = 300
        key = self.key or "0000"
        out_name = f"SAED_Indexing_{key}.svg"
        save_dir = self.image_path.parent if self.image_path is not None else None
        out_default = str((save_dir / out_name) if save_dir is not None else Path(out_name))
        out_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save SVG",
            out_default,
            "SVG (*.svg)",
        )
        if not out_path:
            return

        try:
            import matplotlib.pyplot as plt
            from matplotlib.patches import Arc

            # Make the upper SAED panel closer to square to reduce side white margins.
            fig = plt.figure(figsize=(8.2, 6.2), constrained_layout=True, dpi=dpi)
            gs = fig.add_gridspec(2, 2, width_ratios=[2.8, 1.4], height_ratios=[2.0, 1.0])
            ax_diff = fig.add_subplot(gs[0, 0])
            ax_prof = fig.add_subplot(gs[1, 0])
            ax_tab = fig.add_subplot(gs[:, 1])

            zoom = float(self.spin_zoom.value())
            src = self._image_for_saed_display(self.display_image)
            crop = self._extract_square_crop(src, self.center_xy, zoom)
            clip = float(self.spin_clip.value())
            crop8 = clipped_8bit_image(crop, clip)

            k_scale = self.scale_nm_inv_per_px if self.scale_nm_inv_per_px > 0 else 1.0
            cx_k = (crop.shape[1] * 0.5) * k_scale
            cy_k = (crop.shape[0] * 0.5) * k_scale
            extent = (-cx_k, cx_k, cy_k, -cy_k)
            cmap_name = self.combo_cmap.currentText() if hasattr(self, "combo_cmap") else "gray"
            ax_diff.imshow(crop8, cmap=cmap_name, extent=extent, interpolation="nearest")
            ax_diff.set_title(f"SAED: {key}")
            ax_diff.set_xlabel(r"$k_x$ (nm$^{-1}$)")
            ax_diff.set_ylabel(r"$k_y$ (nm$^{-1}$)")
            ax_diff.set_aspect("equal", adjustable="box")
            ax_diff.plot(0.0, 0.0, marker="+", markersize=7, markeredgewidth=1.0, color="yellow")
            ticks_xy = self._ticks_125(float(extent[0]), float(extent[1]), target_n=6)
            if ticks_xy:
                ax_diff.set_xticks(ticks_xy)
                ax_diff.set_yticks(ticks_xy)

            y_plot = self.profile_sub if self.profile_sub.size else self.profile_i_raw
            if self.chk_log.isChecked() and y_plot.size:
                y_plot = np.log10(np.clip(y_plot - np.min(y_plot) + 1.0, 1e-9, None))
            ax_prof.plot(self.profile_r, y_plot, c="black", lw=0.9)
            ax_prof.set_xlabel(r"$q$ (nm$^{-1}$)")
            ax_prof.set_ylabel("Intensity (log10)" if self.chk_log.isChecked() else "Intensity")
            if self.profile_r.size:
                x_min, x_max = self.plot_x_bar.values() if self._plot_x_initialized else (
                    float(np.min(self.profile_r)),
                    float(np.max(self.profile_r)),
                )
                ax_prof.set_xlim(x_min, x_max)
                mask = (self.profile_r >= x_min) & (self.profile_r <= x_max) & np.isfinite(y_plot)
                if np.any(mask):
                    ys = y_plot[mask]
                else:
                    ys = y_plot[np.isfinite(y_plot)]
                if ys.size > 0:
                    y0 = float(np.percentile(ys, 0.5))
                    y1 = float(np.percentile(ys, 99.5))
                    if y1 <= y0:
                        c = 0.5 * (y0 + y1)
                        y0, y1 = c - 0.5, c + 0.5
                    pad = max(0.02 * (y1 - y0), 1e-9)
                    ax_prof.set_ylim(y0 - pad, y1 + pad)

            colors = [self._pick_qcolor(i, len(self.picks)).getRgbF()[:3] for i in range(max(1, len(self.picks)))]
            for i, p in enumerate(self.picks):
                q = float(p["q"])
                ax_prof.axvline(q, color=colors[i], ls="--", lw=1.2)
                if str(p.get("mode", self._crystal_mode)) == "single":
                    kx = float(p.get("kx", 0.0))
                    ky = float(p.get("ky", 0.0))
                    ax_diff.plot(
                        [kx], [ky],
                        marker="o", markersize=4.2,
                        markerfacecolor="none",
                        markeredgecolor=colors[i],
                        markeredgewidth=1.0,
                        linestyle="None",
                    )
                    ax_diff.text(kx, ky, f"{i+1}", color=colors[i], fontsize=8, ha="left", va="bottom")
                else:
                    ax_diff.add_patch(
                        Arc(
                            (0, 0),
                            width=2 * q,
                            height=2 * q,
                            theta1=0,
                            theta2=90,
                            fill=False,
                            lw=1.2,
                            ls="--",
                            ec=colors[i],
                        )
                    )

            ax_tab.axis("off")
            rows = []
            for i, p in enumerate(self.picks, start=1):
                ang = p.get("angle_deg", float("nan"))
                ang_str = f"{float(ang):.2f}" if np.isfinite(float(ang)) else "—"
                rows.append([str(i), f"{p['q']:.4f}", f"{p['d_ang']:.4f}", ang_str])
            if not rows:
                rows = [["-", "-", "-", "-"]]
            table = ax_tab.table(
                cellText=rows,
                colLabels=["#", "q (nm^-1)", "d (A)", "∠(deg)"],
                loc="upper center",
                cellLoc="center",
            )
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1.1, 1.3)
            # Three-line table: top rule, header-bottom rule, bottom rule.
            n_rows = len(rows)
            n_cols = 4
            for (r, c), cell in table.get_celld().items():
                cell.visible_edges = ""
                cell.set_linewidth(0.0)
            for c in range(n_cols):
                hcell = table[(0, c)]
                hcell.visible_edges = "TB"
                hcell.set_linewidth(0.8)
            for c in range(n_cols):
                bcell = table[(n_rows, c)]
                bcell.visible_edges = "B"
                bcell.set_linewidth(0.8)
            for ridx in range(len(rows)):
                table[(ridx + 1, 0)].get_text().set_fontweight("bold")
                if ridx < len(colors):
                    table[(ridx + 1, 0)].get_text().set_color(colors[ridx])
            ax_tab.set_title("Indexed Planes", pad=2)

            fig.savefig(out_path, format="svg", dpi=dpi)
            plt.close(fig)

            meta_json = json.dumps(self._metadata_dict(), ensure_ascii=False)
            with open(out_path, "r", encoding="utf-8") as f:
                svg = f.read()
            if "<metadata>" in svg:
                import re
                svg = re.sub(r"<metadata>.*?</metadata>", f"<metadata>{meta_json}</metadata>", svg, flags=re.DOTALL)
            else:
                idx = svg.find(">") + 1
                svg = svg[:idx] + f"\n<metadata>{meta_json}</metadata>\n" + svg[idx:]
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(svg)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Save failed", str(e))
            return

        QtWidgets.QMessageBox.information(self, "Saved", f"Saved:\n{out_path}")

    def showEvent(self, event):
        super().showEvent(event)
        if self.profile_r.size > 0 and not self._bg_initialized:
            self._set_default_bg_regions()
            self._bg_initialized = True


def main():
    app = QtWidgets.QApplication.instance()
    owns = False
    if app is None:
        app = QtWidgets.QApplication([])
        owns = True
    win = SAEDIndexingWindow()
    win.show()
    if owns:
        app.exec_()
    return win


if __name__ == "__main__":
    main()
    
