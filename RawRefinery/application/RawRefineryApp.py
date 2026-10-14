import sys
import os
from pathlib import Path
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton,
    QFileDialog, QListWidget, QListWidgetItem, QLabel, QSlider, QSpinBox,
    QHBoxLayout, QFormLayout, QComboBox, QProgressBar, QMessageBox, QSizePolicy,
    QStackedWidget, QGroupBox, QCheckBox, QLineEdit, QAbstractItemView
)
from PySide6.QtGui import QPixmap
from PySide6.QtCore import Qt, Slot, QTimer

# Import utils
from RawRefinery.application.viewing_utils import numpy_to_qimage_rgb
from RawRefinery.application.ModelHandler import ModelController, MODEL_REGISTRY
from RawRefinery.application.ClickableImageLabel import ClickableImageLabel
from RawRefinery.application.dng_utils import to_dng, convert_ccm_to_rational
from RawRefinery.application.LogarithmicSlider import LogarithmicSlider
from RawRefinery.application.LogPanel import LogPanel

RAW_EXTENSIONS = ('.cr2', '.cr3', '.nef', '.arw', '.dng')
PRIMARY_BUTTON_STYLE = (
    "QPushButton { background-color: #0a84ff; color: white; font-weight: 600;"
    " border: none; border-radius: 6px; padding: 8px; }"
    "QPushButton:hover { background-color: #3a9bff; }"
    "QPushButton:disabled { background-color: #3a3a3c; color: #8e8e93; }"
)


def slider_row(slider, spin):
    """Slider and its numeric value on a single row."""
    spin.setFixedWidth(84)
    row = QWidget()
    layout = QHBoxLayout(row)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.addWidget(slider, 1)
    layout.addWidget(spin)
    return row


def explain_open_error(error):
    text = str(error)
    if "Unsupported file format" in text:
        return ("not an original raw file (e.g. a DNG already developed by Lightroom, like an HDR or "
                "panorama merge). RawRefinery needs the camera's original raw data")
    return text


class RawRefineryApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Raw Refinery")
        self.resize(1000, 700)
        # Never start taller/wider than the usable screen area (menu bar and Dock excluded)
        screen = QApplication.primaryScreen()
        if screen is not None:
            avail = screen.availableGeometry()
            self.resize(min(1000, int(avail.width() * 0.9)), min(700, avail.height() - 60))

        #  Logic Controller
        self.controller = ModelController()

        # Connect Signals
        self.controller.progress_update.connect(self.update_progress)
        self.controller.preview_ready.connect(self.display_result)
        self.controller.save_finished.connect(self.reset_after_save)
        self.controller.error_occurred.connect(self.show_error)
        self.controller.model_loaded.connect(lambda n: self.status_label.setText(f"Model Loaded: {n}"))
        self.setup_ui()

        # Load default model
        self.controller.load_model("Tree Net Denoise")

    def loading_popup(self):
        popup = QMessageBox()
        popup.setWindowTitle("Notice")
        popup.setIcon(QMessageBox.Icon.Information)
        popup.setText(
            """
            <h3>Alpha Release Notice</h3>
            <p>This software is currently in Alpha. Please note:</p>
            <ul>
                <li>Expect memory usage up to <b>3 GB</b>.</li>
                <li>CPU processing is slow. Use <b>MPS</b> or <b>CUDA</b> for best results.</li>
            </ul>
            """
        )
        popup.exec()

    # ------------------------------------------------------------------ UI
    def setup_ui(self):
        ## State
        self.dims = None
        self.current_folder = None
        self.current_file_path = None
        self.batch_mode = False
        self.batch_running = False
        self._batch = {}
        self._thumb_lin = None      # linear thumbnail of the current file
        self._last_orig = None      # last preview crop: original / denoised (linear)
        self._last_denoised = None
        self._result_pixmap = None

        # Auto preview: wait a moment after the last change before running the model
        self.preview_timer = QTimer(self)
        self.preview_timer.setSingleShot(True)
        self.preview_timer.setInterval(600)
        self.preview_timer.timeout.connect(self.trigger_preview)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)

        self._build_left_panel()
        self._build_right_panel()
        self.main_layout.addWidget(self.left_panel, 1)
        self.main_layout.addWidget(self.right_panel, 2)

    def _build_left_panel(self):
        self.left_panel = QWidget()
        self.left_layout = QVBoxLayout(self.left_panel)

        top = QHBoxLayout()
        self.btn_open_folder = QPushButton("Open Folder")
        self.btn_open_folder.clicked.connect(self.open_folder_dialog)
        self.btn_batch = QPushButton("Batch mode")
        self.btn_batch.setCheckable(True)
        self.btn_batch.setToolTip("Denoise many photos in one go")
        self.btn_batch.toggled.connect(self.toggle_batch_mode)
        top.addWidget(self.btn_open_folder, 1)
        top.addWidget(self.btn_batch)

        self.file_list = QListWidget()
        self.file_list.currentItemChanged.connect(self.on_file_selected)
        self.file_list.itemChanged.connect(self.update_batch_count)

        # Selection helpers, only visible in batch mode
        self.batch_buttons = QWidget()
        row = QHBoxLayout(self.batch_buttons)
        row.setContentsMargins(0, 0, 0, 0)
        for text, handler in (("Select all", lambda: self.set_all_checked(True)),
                              ("Select none", lambda: self.set_all_checked(False)),
                              ("Check highlighted", self.check_highlighted)):
            button = QPushButton(text)
            button.clicked.connect(handler)
            row.addWidget(button)
        self.batch_buttons.setVisible(False)

        self.left_layout.addLayout(top)
        self.left_layout.addWidget(self.batch_buttons)
        self.left_layout.addWidget(self.file_list)

    def _build_right_panel(self):
        self.right_panel = QWidget()
        self.right_layout = QVBoxLayout(self.right_panel)

        # Previews: thumbnail (click to choose a region) and the 100% result
        self.preview_container = QWidget()
        self.preview_layout = QHBoxLayout(self.preview_container)
        self.thumb_label = ClickableImageLabel("Thumbnail", proportional=True)
        self.thumb_label.imageClicked.connect(self.on_thumbnail_click)

        self.preview_label = QLabel("Select a photo, then click the thumbnail to preview a region")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setWordWrap(True)
        self.preview_label.setMinimumSize(300, 200)
        # The displayed pixmap must not set the window's minimum size
        self.preview_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)

        self.btn_compare = QPushButton("Show original")
        self.btn_compare.setCheckable(True)
        self.btn_compare.setEnabled(False)
        self.btn_compare.setToolTip("Switch between the denoised result and the original, at 100%")
        self.btn_compare.toggled.connect(lambda _: self.render_result())

        result_column = QVBoxLayout()
        result_column.addWidget(self.preview_label, 1)
        result_column.addWidget(self.btn_compare)
        self.preview_layout.addWidget(self.thumb_label)
        self.preview_layout.addLayout(result_column)
        self.right_layout.addWidget(self.preview_container, 1)

        # Page 0: single photo, page 1: batch
        self.right_stack = QStackedWidget()
        self.right_stack.addWidget(self._build_single_page())
        self.right_stack.addWidget(self._build_batch_page())
        self.right_layout.addWidget(self.right_stack)

        # Progress and event log
        self.progress_bar = QProgressBar()
        # Terminal-style event log; keeps the history in a file so old messages are not lost
        self.status_label = LogPanel()
        self.status_label.log("Ready")
        self.right_layout.addWidget(self.progress_bar)
        self.right_layout.addWidget(self.status_label)

    def _build_single_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)

        # Noise level (ISO conditioning of the model)
        self.iso_slider = QSlider(Qt.Horizontal)
        self.iso_slider.setRange(0, 65534)
        self.iso_spin = QSpinBox()
        self.iso_spin.setRange(0, 65534)
        self.iso_slider.valueChanged.connect(self.iso_spin.setValue)
        self.iso_spin.valueChanged.connect(self.iso_slider.setValue)
        self.iso_spin.valueChanged.connect(self.schedule_preview)
        self.iso_slider.setToolTip("How much noise the model expects. Defaults to the photo's ISO")
        form.addRow("Noise level (ISO):", slider_row(self.iso_slider, self.iso_spin))

        # Detail blend: share of the original mixed back into the result
        self.grain_slider = QSlider(Qt.Horizontal)
        self.grain_slider.setRange(0, 100)
        self.grain_spin = QSpinBox()
        self.grain_spin.setRange(0, 100)
        self.grain_spin.setSuffix(" %")
        self.grain_slider.valueChanged.connect(self.grain_spin.setValue)
        self.grain_spin.valueChanged.connect(self.grain_slider.setValue)
        self.grain_spin.valueChanged.connect(self.schedule_preview)
        self.grain_slider.setToolTip("0% = fully denoised. Higher values mix the original back in to keep fine grain")
        form.addRow("Detail blend:", slider_row(self.grain_slider, self.grain_spin))

        # Preview brightness (visualization only)
        self.exposure_slider = LogarithmicSlider(0.1, 10.)
        self.exposure_slider.setNaturalValue(1.)
        self.exposure_value = QLabel(f"{self.exposure_slider.get_natural_value():.1f}×")
        self.exposure_value.setFixedWidth(84)
        self.exposure_slider.setToolTip("Only changes how the preview is displayed, not the saved file")
        self.exposure_slider.naturalValueChanged.connect(self.on_exposure_change)
        form.addRow("Preview brightness:", slider_row(self.exposure_slider, self.exposure_value))
        layout.addLayout(form)

        # Main action
        self.btn_save_cfa = QPushButton("Save denoised DNG…")
        self.btn_save_cfa.setStyleSheet(PRIMARY_BUTTON_STYLE)
        self.btn_save_cfa.setMinimumHeight(36)
        self.btn_save_cfa.clicked.connect(self.trigger_save)
        layout.addWidget(self.btn_save_cfa)

        # Rarely needed options
        self.advanced_group = QGroupBox("Advanced")
        self.advanced_group.setCheckable(True)
        self.advanced_group.setChecked(False)
        self.advanced_group.setFlat(True)
        advanced_layout = QVBoxLayout(self.advanced_group)
        advanced_layout.setContentsMargins(0, 0, 0, 0)
        self.advanced_body = QWidget()
        advanced_form = QFormLayout(self.advanced_body)
        advanced_form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        advanced_form.setContentsMargins(0, 0, 0, 0)

        self.model_combo = QComboBox()
        self.model_combo.addItems(MODEL_REGISTRY.keys())
        self.model_combo.currentTextChanged.connect(self.controller.load_model)
        self.model_combo.currentTextChanged.connect(self.schedule_preview)
        advanced_form.addRow("Model:", self.model_combo)

        self.device_combo = QComboBox()
        self.device_combo.addItems(self.controller.devices)
        self.device_combo.currentTextChanged.connect(self.controller.set_device)
        advanced_form.addRow("Device:", self.device_combo)

        self.btn_save_test_patch = QPushButton("Save original 512×512 patch (testing)")
        self.btn_save_test_patch.setToolTip("Saves the selected region of the original raw, not denoised. For debugging")
        self.btn_save_test_patch.clicked.connect(self.trigger_save_test_patch)
        advanced_form.addRow(self.btn_save_test_patch)

        advanced_layout.addWidget(self.advanced_body)
        self.advanced_body.setVisible(False)
        self.advanced_group.toggled.connect(self.advanced_body.setVisible)
        layout.addWidget(self.advanced_group)
        return page

    def _build_batch_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)

        self.batch_count_label = QLabel("Tick the photos to process in the list on the left.")
        layout.addWidget(self.batch_count_label)

        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        out_row = QWidget()
        out_layout = QHBoxLayout(out_row)
        out_layout.setContentsMargins(0, 0, 0, 0)
        self.batch_out_edit = QLineEdit()
        self.batch_out_edit.setPlaceholderText("Output folder")
        self.batch_out_edit.setReadOnly(True)
        btn_out = QPushButton("Choose…")
        btn_out.clicked.connect(self.choose_batch_output)
        out_layout.addWidget(self.batch_out_edit, 1)
        out_layout.addWidget(btn_out)
        form.addRow("Save into:", out_row)

        self.batch_use_file_iso = QCheckBox("Use each photo's own ISO")
        self.batch_use_file_iso.setChecked(True)
        form.addRow("Noise level:", self.batch_use_file_iso)

        self.batch_blend_spin = QSpinBox()
        self.batch_blend_spin.setRange(0, 100)
        self.batch_blend_spin.setSuffix(" %")
        self.batch_blend_spin.setValue(self.grain_spin.value())
        form.addRow("Detail blend:", self.batch_blend_spin)
        layout.addLayout(form)

        note = QLabel("Originals are never touched. Files are named NAME_denoised.dng and existing files are skipped.")
        note.setWordWrap(True)
        layout.addWidget(note)

        buttons = QHBoxLayout()
        self.btn_batch_start = QPushButton("Start batch")
        self.btn_batch_start.setStyleSheet(PRIMARY_BUTTON_STYLE)
        self.btn_batch_start.setMinimumHeight(36)
        self.btn_batch_start.clicked.connect(self.start_batch)
        self.btn_batch_cancel = QPushButton("Cancel")
        self.btn_batch_cancel.setEnabled(False)
        self.btn_batch_cancel.clicked.connect(self.cancel_batch)
        buttons.addWidget(self.btn_batch_start, 1)
        buttons.addWidget(self.btn_batch_cancel)
        layout.addLayout(buttons)
        return page

    # ----------------------------------------------------------- Display
    def on_exposure_change(self, exposure):
        self.exposure_value.setText(f"{exposure:.1f}×")
        self.render_thumbnail()
        self.render_result()

    def render_thumbnail(self):
        if self._thumb_lin is None:
            return
        qimg = numpy_to_qimage_rgb(self._thumb_lin, exposure=self.exposure_slider.get_natural_value())
        self.thumb_label.set_image(qimg)

    def render_result(self):
        source = self._last_orig if self.btn_compare.isChecked() else self._last_denoised
        if source is None:
            return
        qimg = numpy_to_qimage_rgb(source, exposure=self.exposure_slider.get_natural_value())
        pix = QPixmap.fromImage(qimg)
        self.preview_label.setPixmap(
            pix.scaled(self.preview_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.render_result()

    # ------------------------------------------------------------- Files
    def populate_files(self, files):
        self.file_list.blockSignals(True)
        self.file_list.clear()
        for name in files:
            self.file_list.addItem(QListWidgetItem(name))
        self.file_list.blockSignals(False)
        self.apply_batch_flags()

    def open_folder_dialog(self):
        if self.batch_running:
            return
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder:
            self.current_folder = folder
            files = sorted([f for f in os.listdir(folder) if f.lower().endswith(RAW_EXTENSIONS)])
            self.populate_files(files)
            self.status_label.log(f"Opened folder {folder}: {len(files)} raw files")
            if self.batch_mode:
                self.batch_out_edit.setText(os.path.join(folder, "Denoised"))

    def on_file_selected(self, item, _previous=None):
        if not item or self.batch_mode or self.batch_running:
            return
        path = os.path.join(self.current_folder, item.text())
        self.preview_timer.stop()
        try:
            # Load metadata
            self.dims = None
            iso = self.controller.load_rh(path)
        except Exception as e:
            self.current_file_path = None
            self._thumb_lin = self._last_orig = self._last_denoised = None
            self.status_label.log(f"Error: cannot use {item.text()}: {explain_open_error(e)}", "error")
            return

        self.current_file_path = path
        self.iso_spin.setValue(iso)
        self.status_label.log(f"Opened {item.text()} (ISO {iso})")
        self._last_orig = self._last_denoised = None
        self.btn_compare.setChecked(False)
        self.btn_compare.setEnabled(False)
        self.preview_label.setPixmap(QPixmap())
        self.preview_label.setText("Click the thumbnail to preview a region at 100%")
        try:
            self._thumb_lin = self.controller.generate_thumbnail()
            self.render_thumbnail()
        except Exception as e:
            self.show_error(f"Failed to open file: {e}")

    def on_thumbnail_click(self, x_ratio, y_ratio):
        if self.controller.rh is None:
            return
        # Calculate dims based on click ratio and raw size
        rh = self.controller.rh
        H_full, W_full = rh.raw.shape

        # Center of click in raw coords
        c_x = int(x_ratio * W_full)
        c_y = int(y_ratio * H_full)

        w_preview, h_preview = 512, 512

        # Calculate crops
        h_start = max(0, c_y - h_preview//2)
        h_end = min(H_full, h_start + h_preview)
        w_start = max(0, c_x - w_preview//2)
        w_end = min(W_full, w_start + w_preview)

        self.dims = (h_start, h_end, w_start, w_end)
        self.preview_timer.stop()
        self.trigger_preview()

    def disable_ui(self):
        self.btn_save_cfa.setEnabled(False)
        self.thumb_label.setEnabled(False)

    def enable_ui(self):
        self.btn_save_cfa.setEnabled(True)
        self.thumb_label.setEnabled(True)

    def schedule_preview(self, *_):
        """Re-run the preview shortly after a setting changed (only once a region has been chosen)."""
        if self.dims and not self.batch_mode and not self.batch_running:
            self.preview_timer.start()

    def trigger_preview(self):
        if not self.dims:
            self.status_label.setText("Click the thumbnail to choose a region first.")
            return

        conditioning = [self.iso_spin.value(), self.grain_spin.value()]
        # Disable controls to prevent spamming
        self.disable_ui()
        self.status_label.setText("Processing preview...")
        # Fire off the worker
        self.controller.run_inference(conditioning, self.dims)

    def trigger_save(self):
        if not self.current_file_path:
            self.status_label.setText("Select an image first.")
            return

        output_filename, _ = QFileDialog.getSaveFileName(
            self, "Save Denoised Image",
            os.path.splitext(self.current_file_path)[0] + "_denoised.dng",
            "DNG Image (*.dng)"
        )
        if not output_filename:
            return  # dialog cancelled
        if os.path.abspath(output_filename) == os.path.abspath(self.current_file_path):
            self.show_error("Refusing to overwrite the original file. Choose another name.")
            return

        conditioning = [self.iso_spin.value(), self.grain_spin.value()]
        self.disable_ui()
        self.status_label.setText(f"Denoising {os.path.basename(self.current_file_path)} (full image)...")
        self.controller.save_image(output_filename, conditioning, save_cfa=True)

    def trigger_save_test_patch(self):
        if not self.current_file_path or not self.dims:
            self.status_label.setText("Select an image and a region on the thumbnail first.")
            return
        try:
            output_filename, _ = QFileDialog.getSaveFileName(
                self, "Save Original Patch",
                os.path.splitext(self.current_file_path)[0] + "_test_patch.dng",
                "DNG Image (*.dng)"
            )
            if not output_filename:
                return
            cfa = self.controller.rh._input_handler(dims=self.dims)
            cfa = np.squeeze(cfa, axis=0)
            ccm1 = convert_ccm_to_rational(self.controller.rh.core_metadata.rgb_xyz_matrix[:3, :])
            to_dng(cfa, self.controller.rh, output_filename, ccm1, save_cfa=True, convert_to_cfa=False, use_orig_wb_points=True)
            self.status_label.log(f"Saved test patch {os.path.basename(output_filename)}")
        except Exception as e:
            self.show_error(f"Failed to save patch: {e}")

    # -------------------------------------------------------------- Batch
    def toggle_batch_mode(self, enabled):
        if self.batch_running:
            self.btn_batch.setChecked(True)   # can't leave while running
            return
        self.batch_mode = enabled
        self.btn_batch.setText("Exit batch mode" if enabled else "Batch mode")
        self.right_stack.setCurrentIndex(1 if enabled else 0)
        self.preview_container.setVisible(not enabled)
        self.batch_buttons.setVisible(enabled)
        self.file_list.setSelectionMode(
            QAbstractItemView.ExtendedSelection if enabled else QAbstractItemView.SingleSelection)
        if enabled:
            self.preview_timer.stop()
            if self.current_folder and not self.batch_out_edit.text():
                self.batch_out_edit.setText(os.path.join(self.current_folder, "Denoised"))
            self.batch_blend_spin.setValue(self.grain_spin.value())
        self.apply_batch_flags()
        self.status_label.log("Batch mode on: tick the photos to process" if enabled else "Batch mode off")

    def apply_batch_flags(self):
        """Show check boxes next to the files in batch mode, hide them otherwise."""
        self.file_list.blockSignals(True)
        for i in range(self.file_list.count()):
            item = self.file_list.item(i)
            if self.batch_mode:
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(Qt.Unchecked)
            else:
                item.setFlags(item.flags() & ~Qt.ItemIsUserCheckable)
                item.setData(Qt.CheckStateRole, None)
        self.file_list.blockSignals(False)
        self.update_batch_count()

    def checked_files(self):
        return [self.file_list.item(i).text() for i in range(self.file_list.count())
                if self.file_list.item(i).checkState() == Qt.Checked]

    def update_batch_count(self, *_):
        if not self.batch_mode:
            return
        count = len(self.checked_files())
        self.batch_count_label.setText(
            f"{count} of {self.file_list.count()} photos selected" if count
            else "Tick the photos to process in the list on the left.")

    def set_all_checked(self, checked):
        self.file_list.blockSignals(True)
        for i in range(self.file_list.count()):
            self.file_list.item(i).setCheckState(Qt.Checked if checked else Qt.Unchecked)
        self.file_list.blockSignals(False)
        self.update_batch_count()

    def check_highlighted(self):
        self.file_list.blockSignals(True)
        for item in self.file_list.selectedItems():
            item.setCheckState(Qt.Checked)
        self.file_list.blockSignals(False)
        self.update_batch_count()

    def choose_batch_output(self):
        start = self.batch_out_edit.text() or self.current_folder or ""
        folder = QFileDialog.getExistingDirectory(self, "Output folder", start)
        if folder:
            self.batch_out_edit.setText(folder)

    def set_batch_controls_enabled(self, enabled):
        for widget in (self.btn_open_folder, self.btn_batch, self.file_list, self.batch_buttons,
                       self.batch_out_edit, self.batch_use_file_iso, self.batch_blend_spin,
                       self.btn_batch_start):
            widget.setEnabled(enabled)
        self.btn_batch_cancel.setEnabled(not enabled)

    def start_batch(self):
        jobs = self.checked_files()
        if not jobs:
            self.status_label.log("Batch: no photos selected")
            return
        if not self.batch_out_edit.text():
            self.status_label.log("Batch: choose an output folder first", "error")
            return
        out_dir = Path(self.batch_out_edit.text())
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            self.status_label.log(f"Error: cannot create {out_dir}: {e}", "error")
            return
        self._batch = {"jobs": jobs, "index": 0, "ok": 0, "skipped": 0, "failed": 0,
                       "out_dir": out_dir, "cancel": False}
        self.batch_running = True
        self.set_batch_controls_enabled(False)
        self.progress_bar.setValue(0)
        self.status_label.log(f"Batch started: {len(jobs)} photos -> {out_dir}")
        self.batch_next()

    def cancel_batch(self):
        if self.batch_running:
            self._batch["cancel"] = True
            self.btn_batch_cancel.setEnabled(False)
            self.status_label.log("Batch: cancelling after the current photo...")

    def batch_next(self):
        batch = self._batch
        if batch["cancel"] or batch["index"] >= len(batch["jobs"]):
            self.batch_finish()
            return
        name = batch["jobs"][batch["index"]]
        batch["index"] += 1
        tag = f"[{batch['index']}/{len(batch['jobs'])}] {name}"
        out_path = batch["out_dir"] / f"{Path(name).stem}_denoised.dng"
        src_path = os.path.join(self.current_folder, name)

        if out_path.exists():
            batch["skipped"] += 1
            self.status_label.log(f"{tag}: skipped, {out_path.name} already exists")
            QTimer.singleShot(0, self.batch_next)
            return
        try:
            iso = self.controller.load_rh(src_path)
        except Exception as e:
            batch["skipped"] += 1
            self.status_label.log(f"{tag}: skipped, {explain_open_error(e)}", "error")
            QTimer.singleShot(0, self.batch_next)
            return

        iso_value = iso if self.batch_use_file_iso.isChecked() else self.iso_spin.value()
        self.status_label.log(f"{tag}: denoising (ISO {iso_value})...")
        self.controller.save_image(str(out_path), [iso_value, self.batch_blend_spin.value()], save_cfa=True)

    def batch_finish(self):
        batch = self._batch
        self.batch_running = False
        self.set_batch_controls_enabled(True)
        self.progress_bar.setValue(100)
        state = "cancelled" if batch["cancel"] else "finished"
        self.status_label.log(
            f"Batch {state}: {batch['ok']} saved, {batch['skipped']} skipped, {batch['failed']} failed",
            "error" if batch["failed"] else "success")

    # -------------------------------------------------------------- Slots
    @Slot(float)
    def update_progress(self, val):
        if self.batch_running:
            done = self._batch["index"] - 1 + val
            self.progress_bar.setValue(int(done / len(self._batch["jobs"]) * 100))
        else:
            self.progress_bar.setValue(int(val * 100))

    @Slot(object, object)
    def display_result(self, img_rgb, denoised):
        self.enable_ui()
        self.status_label.setText("Preview ready.")
        self.progress_bar.setValue(100)
        self._last_orig = img_rgb
        self._last_denoised = denoised
        self.btn_compare.setEnabled(True)
        self.render_result()

    @Slot(str)
    def reset_after_save(self, message):
        if self.batch_running:
            self._batch["ok"] += 1
            self.status_label.log(message, "success")
            QTimer.singleShot(200, self.batch_next)   # let the finished worker thread clean up first
            return
        self.enable_ui()
        self.status_label.setText(message)
        self.progress_bar.setValue(100)

    @Slot(str)
    def show_error(self, msg):
        if self.batch_running:
            self._batch["failed"] += 1
            self.status_label.log(f"Error: {msg}", "error")
            QTimer.singleShot(200, self.batch_next)
            return
        self.enable_ui()
        self.status_label.log(f"Error: {msg}", "error")
        QMessageBox.critical(self, "Error", msg)

    @Slot(str)
    def display_device(self, device):
        print("device used")
        self.device_label.setText(device)

import platform
if __name__ == '__main__':
    app = QApplication(sys.argv)
    if platform.system() == "Darwin":  # Darwin is the kernel for macOS
        app.setApplicationDisplayName("Raw Refinery")
        app.setOrganizationName("Ryan Mueller")
        app.setDesktopFileName("com.rawrefinery.app")

    window = RawRefineryApp()
    window.show()
    window.loading_popup()
    sys.exit(app.exec())
