import sys
import subprocess
import os
import platform
import yaml
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QTextEdit, QLabel, QSplitter,
    QTabWidget, QFormLayout, QSpinBox, QDoubleSpinBox, QCheckBox,
    QComboBox, QLineEdit, QScrollArea, QGroupBox, QRadioButton, QMessageBox
)
from PyQt6.QtCore import QThread, pyqtSignal, Qt
from PyQt6.QtGui import QPixmap


def load_config(filename):
    """Loads a YAML config file and returns a dictionary."""
    try:
        with open(filename, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Warning: Config file '{filename}' not found. Using hardcoded defaults.")
        return {}
    except Exception as e:
        print(f"Warning: Error loading '{filename}': {e}. Using hardcoded defaults.")
        return {}

class RunThread(QThread):
    output = pyqtSignal(str)
    
    def __init__(self, cmd_list):
        super().__init__()
        self.cmd_list = cmd_list

    def run(self):
        try:
            is_windows = platform.system() == "Windows"
            self.output.emit(f"Running command: {' '.join(self.cmd_list)}\n")
            
            process = subprocess.run(
                self.cmd_list,
                capture_output=True,
                text=True,
                check=True,
                encoding='cp1252' if is_windows else 'utf-8',
                shell=is_windows
            )
            self.output.emit(process.stdout)
        except subprocess.CalledProcessError as e:
            error_message = f"--- ERROR ---\n{e.stderr}\n--- STDOUT ---\n{e.stdout}"
            self.output.emit(error_message)
        except FileNotFoundError:
            self.output.emit(f"Error: Script not found. Make sure {self.cmd_list[1]} is in the same directory.")
        except Exception as e:
            self.output.emit(f"An unexpected error occurred: {str(e)}")

class PathSelector(QWidget):
    def __init__(self, is_directory=False):
        super().__init__()
        self.is_directory = is_directory
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.line_edit = QLineEdit()
        layout.addWidget(self.line_edit)
        
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self.browse)
        layout.addWidget(browse_btn)

    def browse(self):
        if self.is_directory:
            path = QFileDialog.getExistingDirectory(self, "Select Directory")
        else:
            path, _ = QFileDialog.getOpenFileName(self, "Select File", "", "All Files (*)")
        
        if path:
            self.line_edit.setText(path)
            
    def text(self):
        return self.line_edit.text()
    
    def setText(self, text):
        self.line_edit.setText(text)

def create_radio_group(title, options, default_key='default'):
    group_box = QGroupBox(title)
    group_layout = QVBoxLayout()
    
    buttons = {}
    for key, text in options.items():
        radio = QRadioButton(text)
        if key == default_key:
            radio.setChecked(True) 
        group_layout.addWidget(radio)
        buttons[key] = radio
        
    group_box.setLayout(group_layout)
    return group_box, buttons

class ScopeXRApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SCOPE-XR GUI")
        self.resize(1100, 800)
        
        self.image_path = None

        try:
            self.base_dir = os.path.dirname(os.path.abspath(__file__))
        except NameError:
            self.base_dir = os.path.abspath(".")
            
        fs_config_path = os.path.join(self.base_dir, 'fs_args.yaml')
        psf_config_path = os.path.join(self.base_dir, 'psf_args.yaml')

        self.fs_config_data = load_config(fs_config_path)
        self.psf_config_data = load_config(psf_config_path)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.setCentralWidget(splitter)

        left_pane = QWidget()
        left_layout = QVBoxLayout(left_pane)
        
        self.load_image_btn = QPushButton("Load Image...")
        left_layout.addWidget(self.load_image_btn)

        self.image_display_label = QLabel("Load an image to see a preview")
        self.image_display_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_display_label.setMinimumSize(400, 400)
        self.image_display_label.setStyleSheet(
            "border: 1px dashed #aaa; background-color: #f0f0f0;"
        )
        left_layout.addWidget(self.image_display_label, stretch=1)
        splitter.addWidget(left_pane)

        right_pane = QWidget()
        right_layout = QVBoxLayout(right_pane)

        self.tab_widget = QTabWidget()
        
        self.fs_tab = self.create_fs_tab(self.fs_config_data)
        self.psf_tab = self.create_psf_tab(self.psf_config_data)
        
        self.tab_widget.addTab(self.fs_tab, "Focal Spot (FS)")
        self.tab_widget.addTab(self.psf_tab, "PSF")
        
        right_layout.addWidget(self.tab_widget)

        button_layout = QHBoxLayout()
        self.edit_config_btn = QPushButton("Edit Default Config File")
        self.run_btn = QPushButton("Run Analysis")
        self.run_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        
        button_layout.addWidget(self.edit_config_btn)
        button_layout.addWidget(self.run_btn)
        right_layout.addLayout(button_layout)

        right_layout.addWidget(QLabel("Output:"))
        self.output_console = QTextEdit()
        self.output_console.setReadOnly(True)
        self.output_console.setPlaceholderText("Script output will appear here...")
        right_layout.addWidget(self.output_console, stretch=1)

        splitter.addWidget(right_pane)
        splitter.setSizes([500, 600])

        self.load_image_btn.clicked.connect(self.open_image_file)
        self.edit_config_btn.clicked.connect(self.edit_config)
        self.run_btn.clicked.connect(self.run_script)
        
        self.run_thread = None

    def _create_scrollable_tab(self):
        tab_widget = QWidget()
        tab_layout = QVBoxLayout(tab_widget)
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_content = QWidget()
        form_layout = QFormLayout(scroll_content)
        scroll_area.setWidget(scroll_content)
        tab_layout.addWidget(scroll_area)
        return tab_widget, form_layout

    # --- UPDATE FUNCTION FOR FS TAB ---
    def update_fs_gui(self):
        """Reads the file path from the FS config box, loads it, and updates widgets."""
        path = self.fs_config.text()
        if not path or not os.path.exists(path):
            QMessageBox.warning(self, "File Not Found", f"Could not find config file: {path}")
            return

        new_data = load_config(path)
        if not new_data:
            QMessageBox.warning(self, "Error", "Failed to load or empty config file.")
            return

        # Update widgets (use .get() with a safe fallback or keep current value)
        # We use a simple helper to avoid clutter
        def set_spin(widget, key, dtype=float):
            if key in new_data:
                val = new_data[key]
                if val is not None:
                    widget.setValue(dtype(val))

        if 'out_dir' in new_data: self.fs_output_dir.setText(new_data['out_dir'])
        set_spin(self.fs_pixel_size, 'pixel_size', float)
        set_spin(self.fs_diameter, 'circle_diameter', float)
        if 'no_hough' in new_data: self.fs_no_hough.setChecked(bool(new_data['no_hough']))
        set_spin(self.fs_magnification, 'm', float)
        set_spin(self.fs_min_pixels, 'n', int)
        set_spin(self.fs_nangles, 'n_angles', int)
        set_spin(self.fs_half_length, 'profile_half_length', int)
        set_spin(self.fs_derivative_step, 'derivative_step', int)
        set_spin(self.fs_axis_shifts, 'axis_shifts', int)
        
        if 'filter_name' in new_data: 
            self.fs_filter.setCurrentText(str(new_data['filter_name']))
        set_spin(self.fs_avg_number, 'avg_number', int)
        if 'symmetrize' in new_data: self.fs_sym.setChecked(bool(new_data['symmetrize']))
        
        # Shifts
        manual_val = new_data.get('manual_shift')
        if manual_val is not None:
            self.fs_radio_manual.setChecked(True)
            self.fs_manual_shift_val.setValue(int(manual_val))
        elif new_data.get('no_shift', False):
            self.fs_radio_no.setChecked(True)
        elif new_data.get('auto_shift', False): # Explicit check
             self.fs_radio_auto.setChecked(True)
        else:
             # Fallback to default if nothing specified? Or leave as is?
             # Let's assume auto_shift is default if nothing else is set
             self.fs_radio_auto.setChecked(True)

        # Avg Group
        if new_data.get('avg_neighbors', False):
            self.fs_avg_buttons['avg'].setChecked(True)
        elif new_data.get('no_avg', False): # Assuming you add this key to yaml for explicit disable
             self.fs_avg_buttons['no_avg'].setChecked(True)
        # 'default' button stays if neither key is present

        if 'show_plots' in new_data: self.fs_show.setChecked(bool(new_data['show_plots']))
        
        print(f"FS GUI updated from {path}")


    def create_fs_tab(self, config_data):
        tab_widget, layout = self._create_scrollable_tab()
        
        # Config File Row with Load Button
        config_layout = QHBoxLayout()
        self.fs_config = PathSelector(is_directory=False)
        # Assuming config_data has the default path or we use the hardcoded one
        # self.fs_config.setText(...) # Set initial text if you have it
        
        load_btn = QPushButton("Load/Update GUI")
        load_btn.clicked.connect(self.update_fs_gui)
        
        config_layout.addWidget(self.fs_config)
        config_layout.addWidget(load_btn)
        layout.addRow("Config File [--config]:", config_layout)

        self.fs_output_dir = PathSelector(is_directory=True)
        self.fs_output_dir.setText(config_data.get('out_dir', ''))
        layout.addRow("Output Dir [--o]:", self.fs_output_dir)
        
        self.fs_pixel_size = QDoubleSpinBox()
        self.fs_pixel_size.setDecimals(10)
        self.fs_pixel_size.setValue(config_data.get('pixel_size', 0.1))
        layout.addRow("Pixel Size (mm) [--p]:", self.fs_pixel_size)

        self.fs_diameter = QDoubleSpinBox()
        self.fs_diameter.setDecimals(2)
        self.fs_diameter.setRange(0.01, 1000.0)
        self.fs_diameter.setValue(config_data.get('circle_diameter', 1.0))
        layout.addRow("Object Diameter (mm) [--d]:", self.fs_diameter)

        self.fs_no_hough = QCheckBox("Skip Hough Transform")
        self.fs_no_hough.setChecked(config_data.get('no_hough', False))
        layout.addRow("[--no_hough]:", self.fs_no_hough)

        self.fs_magnification = QDoubleSpinBox()
        self.fs_magnification.setDecimals(3)
        self.fs_magnification.setValue(config_data.get('m', 0.0))
        self.fs_magnification.setToolTip("Set to 0.0 to let program estimate automatically.")
        layout.addRow("Magnification [--m]:", self.fs_magnification)
        
        self.fs_min_pixels = QSpinBox()
        self.fs_min_pixels.setRange(0, 500)
        self.fs_min_pixels.setValue(config_data.get('n', 10))
        layout.addRow("Min. Pixels [--n]:", self.fs_min_pixels)
        
        self.fs_nangles = QSpinBox()
        self.fs_nangles.setRange(90, 1080)
        self.fs_nangles.setValue(config_data.get('n_angles', 360)) 
        layout.addRow("Num. Angles [--nangles]:", self.fs_nangles)
        
        self.fs_half_length = QSpinBox()
        self.fs_half_length.setRange(1, 10_000)
        self.fs_half_length.setValue(config_data.get('profile_half_length', 100)) 
        layout.addRow("Profile Half-Length [--hl]:", self.fs_half_length)

        self.fs_derivative_step = QSpinBox()
        self.fs_derivative_step.setRange(1, 10)
        self.fs_derivative_step.setValue(config_data.get('derivative_step', 1)) 
        layout.addRow("Derivative Step [--ds]:", self.fs_derivative_step)

        self.fs_axis_shifts = QSpinBox()
        self.fs_axis_shifts.setRange(0, 50)
        self.fs_axis_shifts.setValue(config_data.get('axis_shifts', 10)) 
        layout.addRow("Shifts Search Range [--axis_shifts]:", self.fs_axis_shifts)
        
        self.fs_filter = QComboBox()
        self.fs_filter.addItems(["ramp", "shepp-logan", "cosine", "hamming", "hann", "None"])
        self.fs_filter.setCurrentText(str(config_data.get('filter_name', 'ramp'))) 
        layout.addRow("Filter [--filter]:", self.fs_filter)

        self.fs_avg_number = QSpinBox()
        self.fs_avg_number.setRange(1, 99); self.fs_avg_number.setSingleStep(2)
        self.fs_avg_number.setValue(config_data.get('avg_number', 3))
        layout.addRow("Avg. Number (must be odd) [--avg_number]:", self.fs_avg_number)
        
        self.fs_sym = QCheckBox("Symmetrize Sinogram")
        self.fs_sym.setChecked(config_data.get('symmetrize', False)) 
        layout.addRow("[--sym]:", self.fs_sym)
        
        fs_shift_box = QGroupBox("Sinogram Shifting")
        fs_shift_layout = QVBoxLayout()
        self.fs_radio_auto = QRadioButton("Auto Shift (--auto_shift)")
        self.fs_radio_no = QRadioButton("No Shift (--no_shift)")
        
        self.fs_radio_manual = QRadioButton("Manual Shift (--manual_shift)")
        self.fs_manual_shift_val = QSpinBox()
        self.fs_manual_shift_val.setRange(-1000, 1000)
        self.fs_manual_shift_val.setEnabled(False) 
        self.fs_radio_manual.toggled.connect(self.fs_manual_shift_val.setEnabled)
        
        manual_layout = QHBoxLayout()
        manual_layout.addWidget(self.fs_radio_manual)
        manual_layout.addWidget(self.fs_manual_shift_val)
        
        fs_shift_layout.addWidget(self.fs_radio_auto)
        fs_shift_layout.addLayout(manual_layout)
        fs_shift_layout.addWidget(self.fs_radio_no)
        fs_shift_box.setLayout(fs_shift_layout)

        manual_val = config_data.get('manual_shift')
        if manual_val is not None:
            self.fs_radio_manual.setChecked(True)
            self.fs_manual_shift_val.setValue(int(manual_val))
        elif config_data.get('no_shift', False):
            self.fs_radio_no.setChecked(True)
        else:
            self.fs_radio_auto.setChecked(True) 
            
        layout.addRow(fs_shift_box)

        avg_default = 'default'
        if config_data.get('avg_neighbors', False): avg_default = 'avg' 
        elif config_data.get('no_avg', False): avg_default = 'no_avg'
        
        self.fs_avg_group, self.fs_avg_buttons = create_radio_group(
            "Profile Averaging",
            {"default": "Default (from YAML)", "avg": "Enable (--avg)", "no_avg": "Disable (--no_avg)"},
            default_key=avg_default
        )
        layout.addRow(self.fs_avg_group)

        self.fs_show = QCheckBox("Show Matplotlib plots")
        self.fs_show.setChecked(config_data.get('show_plots', True)) 
        layout.addRow("[--show]:", self.fs_show)
        
        return tab_widget

    # --- UPDATE FUNCTION FOR PSF TAB ---
    def update_psf_gui(self):
        """Reads the file path from the PSF config box, loads it, and updates widgets."""
        path = self.psf_config.text()
        if not path or not os.path.exists(path):
            QMessageBox.warning(self, "File Not Found", f"Could not find config file: {path}")
            return

        new_data = load_config(path)
        if not new_data:
            QMessageBox.warning(self, "Error", "Failed to load or empty config file.")
            return

        def set_spin(widget, key, dtype=float):
            if key in new_data:
                val = new_data[key]
                if val is not None:
                    widget.setValue(dtype(val))

        if 'out_dir' in new_data: self.psf_output_dir.setText(new_data['out_dir'])
        set_spin(self.psf_pixel_size, 'pixel_size', float)
        set_spin(self.psf_diameter, 'circle_diameter', float)
        if 'no_hough' in new_data: self.psf_no_hough.setChecked(bool(new_data['no_hough']))
        set_spin(self.psf_nangles, 'n_angles', int)
        set_spin(self.psf_half_length, 'profile_half_length', int)
        set_spin(self.psf_derivative_step, 'derivative_step', int)
        set_spin(self.psf_axis_shifts, 'axis_shifts', int)
        
        if 'filter_name' in new_data: 
            self.psf_filter.setCurrentText(str(new_data['filter_name']))
        set_spin(self.psf_avg_number, 'avg_number', int)
        if 'symmetrize' in new_data: self.psf_sym.setChecked(bool(new_data['symmetrize']))
        
        set_spin(self.psf_dtheta, 'dtheta', float)
        set_spin(self.psf_resample1, 'resample1', float)
        set_spin(self.psf_resample2, 'resample2', float)
        set_spin(self.psf_gaussian_sigma, 'gaussian_sigma', float)
        if 'oversample_strategy' in new_data:
            self.psf_oversample_strategy.setCurrentText(str(new_data['oversample_strategy']))

        # Shifts
        manual_val = new_data.get('manual_shift')
        if manual_val is not None:
            self.psf_radio_manual.setChecked(True)
            self.psf_manual_shift_val.setValue(int(manual_val))
        elif new_data.get('no_shift', False):
            self.psf_radio_no.setChecked(True)
        elif new_data.get('auto_shift', False):
             self.psf_radio_auto.setChecked(True)
        else:
             self.psf_radio_auto.setChecked(True)

        # Avg Group
        if new_data.get('avg_neighbors', False):
            self.psf_avg_buttons['avg'].setChecked(True)
        elif new_data.get('no_avg', False): 
             self.psf_avg_buttons['no_avg'].setChecked(True)

        # Oversample Group
        if new_data.get('oversample', False):
            self.psf_oversample_buttons['oversample'].setChecked(True)
        elif new_data.get('no_oversample', False):
             self.psf_oversample_buttons['no_oversample'].setChecked(True)

        if 'show_plots' in new_data: self.psf_show.setChecked(bool(new_data['show_plots']))
        
        print(f"PSF GUI updated from {path}")

    def create_psf_tab(self, config_data):
        tab_widget, layout = self._create_scrollable_tab()

        # Config File Row with Load Button
        config_layout = QHBoxLayout()
        self.psf_config = PathSelector(is_directory=False)
        
        load_btn = QPushButton("Load/Update GUI")
        load_btn.clicked.connect(self.update_psf_gui)
        
        config_layout.addWidget(self.psf_config)
        config_layout.addWidget(load_btn)
        layout.addRow("Config File [--config]:", config_layout)

        self.psf_output_dir = PathSelector(is_directory=True)
        self.psf_output_dir.setText(config_data.get('out_dir', '')) 
        layout.addRow("Output Dir [--o]:", self.psf_output_dir)
        
        self.psf_pixel_size = QDoubleSpinBox()
        self.psf_pixel_size.setDecimals(10)
        self.psf_pixel_size.setValue(config_data.get('pixel_size', 0.1)) 
        layout.addRow("Pixel Size (mm) [--p]:", self.psf_pixel_size)

        self.psf_diameter = QDoubleSpinBox()
        self.psf_diameter.setDecimals(2)
        self.psf_diameter.setRange(0.01, 1000.0)
        self.psf_diameter.setValue(config_data.get('circle_diameter', 1.0)) 
        layout.addRow("Object Diameter (mm) [--d]:", self.psf_diameter)

        self.psf_no_hough = QCheckBox("Skip Hough Transform")
        self.psf_no_hough.setChecked(config_data.get('no_hough', False))
        layout.addRow("[--no_hough]:", self.psf_no_hough)

        self.psf_nangles = QSpinBox()
        self.psf_nangles.setRange(90, 1080)
        self.psf_nangles.setValue(config_data.get('n_angles', 360)) 
        layout.addRow("Num. Angles [--nangles]:", self.psf_nangles)
        
        self.psf_half_length = QSpinBox()
        self.psf_half_length.setRange(1, 10_000)
        self.psf_half_length.setValue(config_data.get('profile_half_length', 100)) 
        layout.addRow("Profile Half-Length [--hl]:", self.psf_half_length)

        self.psf_derivative_step = QSpinBox()
        self.psf_derivative_step.setRange(1, 10)
        self.psf_derivative_step.setValue(config_data.get('derivative_step', 1)) 
        layout.addRow("Derivative Step [--ds]:", self.psf_derivative_step)

        self.psf_axis_shifts = QSpinBox()
        self.psf_axis_shifts.setRange(0, 50)
        self.psf_axis_shifts.setValue(config_data.get('axis_shifts', 10)) 
        layout.addRow("Shifts Search Range [--axis_shifts]:", self.psf_axis_shifts)
        
        self.psf_filter = QComboBox()
        self.psf_filter.addItems(["ramp", "shepp-logan", "cosine", "hamming", "hann", "None"])
        self.psf_filter.setCurrentText(str(config_data.get('filter_name', 'ramp'))) 
        layout.addRow("Filter [--filter]:", self.psf_filter)

        self.psf_avg_number = QSpinBox()
        self.psf_avg_number.setRange(1, 99); self.psf_avg_number.setSingleStep(2)
        self.psf_avg_number.setValue(config_data.get('avg_number', 3))
        layout.addRow("Avg. Number (must be odd) [--avg_number]:", self.psf_avg_number)
        
        self.psf_sym = QCheckBox("Symmetrize Sinogram")
        self.psf_sym.setChecked(config_data.get('symmetrize', False)) 
        layout.addRow("[--sym]:", self.psf_sym)

        self.psf_dtheta = QDoubleSpinBox()
        self.psf_dtheta.setDecimals(2)
        self.psf_dtheta.setValue(config_data.get('dtheta', 10.0))
        layout.addRow("Oversample Angle (deg) [--dtheta]:", self.psf_dtheta)

        self.psf_resample1 = QDoubleSpinBox()
        self.psf_resample1.setValue(config_data.get('resample1', 5))
        layout.addRow("Resample 1 (fine) [--resample1]:", self.psf_resample1)

        self.psf_resample2 = QDoubleSpinBox()
        self.psf_resample2.setValue(config_data.get('resample2', 2))
        layout.addRow("Resample 2 (coarse) [--resample2]:", self.psf_resample2)

        self.psf_gaussian_sigma = QDoubleSpinBox()
        self.psf_gaussian_sigma.setDecimals(2)
        self.psf_gaussian_sigma.setValue(config_data.get('gaussian_sigma', 1.0))
        layout.addRow("Gaussian Sigma [--gaussian_sigma]:", self.psf_gaussian_sigma)

        self.psf_oversample_strategy = QComboBox()
        self.psf_oversample_strategy.addItems(["1", "2"])
        self.psf_oversample_strategy.setCurrentText(str(config_data.get('oversample_strategy', '1')))
        layout.addRow("Oversample Strategy:", self.psf_oversample_strategy)
        
        # PSF Shifts
        psf_shift_box = QGroupBox("Sinogram Shifting")
        psf_shift_layout = QVBoxLayout()
        self.psf_radio_auto = QRadioButton("Auto Shift (--auto_shift)")
        self.psf_radio_no = QRadioButton("No Shift (--no_shift)")
        
        self.psf_radio_manual = QRadioButton("Manual Shift (--manual_shift)")
        self.psf_manual_shift_val = QSpinBox()
        self.psf_manual_shift_val.setRange(-1000, 1000)
        self.psf_manual_shift_val.setEnabled(False) 
        self.psf_radio_manual.toggled.connect(self.psf_manual_shift_val.setEnabled)
        
        manual_layout_psf = QHBoxLayout()
        manual_layout_psf.addWidget(self.psf_radio_manual)
        manual_layout_psf.addWidget(self.psf_manual_shift_val)
        
        psf_shift_layout.addWidget(self.psf_radio_auto)
        psf_shift_layout.addLayout(manual_layout_psf)
        psf_shift_layout.addWidget(self.psf_radio_no)
        psf_shift_box.setLayout(psf_shift_layout)

        manual_val_psf = config_data.get('manual_shift')
        if manual_val_psf is not None:
            self.psf_radio_manual.setChecked(True)
            self.psf_manual_shift_val.setValue(int(manual_val_psf))
        elif config_data.get('no_shift', False):
            self.psf_radio_no.setChecked(True)
        else:
            self.psf_radio_auto.setChecked(True)
            
        layout.addRow(psf_shift_box)

        avg_default = 'default'
        if config_data.get('avg_neighbors', False): avg_default = 'avg' 
        elif config_data.get('no_avg', False): avg_default = 'no_avg'

        self.psf_avg_group, self.psf_avg_buttons = create_radio_group(
            "Profile Averaging",
            {"default": "Default (from YAML)", "avg": "Enable (--avg)", "no_avg": "Disable (--no_avg)"},
            default_key=avg_default
        )
        layout.addRow(self.psf_avg_group)

        oversample_default = 'default'
        if config_data.get('oversample', False): oversample_default = 'oversample'
        elif config_data.get('no_oversample', False): oversample_default = 'no_oversample'

        self.psf_oversample_group, self.psf_oversample_buttons = create_radio_group(
            "Oversampling",
            {"default": "Default (from YAML)", "oversample": "Enable (--oversample)", "no_oversample": "Disable (--no_oversample)"},
            default_key=oversample_default
        )
        layout.addRow(self.psf_oversample_group)
        
        self.psf_show = QCheckBox("Show Matplotlib plots")
        self.psf_show.setChecked(config_data.get('show_plots', True)) 
        layout.addRow("[--show]:", self.psf_show)
        
        return tab_widget

    def open_image_file(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "", "Image Files (*.png *.tif *.raw)"
        )
        if file_name:
            self.image_path = file_name
            if file_name.endswith(".raw"):
                self.image_display_label.setText(f"RAW file selected:\n{os.path.basename(file_name)}\n(Preview not available)")
                self.image_display_label.setStyleSheet("")
            else:
                pixmap = QPixmap(self.image_path)
                self.update_image_display(pixmap)
                self.image_display_label.setStyleSheet("")

    def update_image_display(self, pixmap):
        scaled_pixmap = pixmap.scaled(
            self.image_display_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.image_display_label.setPixmap(scaled_pixmap)

    def resizeEvent(self, event):
        if self.image_path and not self.image_path.endswith(".raw"):
            self.update_image_display(QPixmap(self.image_path))
        super().resizeEvent(event)

    def edit_config(self):
        current_tab_index = self.tab_widget.currentIndex()
        config_filename = "fs_args.yaml" if current_tab_index == 0 else "psf_args.yaml"
        
        config_file_path = os.path.join(self.base_dir, config_filename)
        try:
            system = platform.system()
            if system == "Windows":
                os.startfile(config_file_path)
            elif system == "Darwin":
                subprocess.run(["open", config_file_path])
            else:
                subprocess.run(["xdg-open", config_file_path])
        except Exception as e:
            self.output_console.setText(f"Error opening {config_file_path}: {str(e)}")

    def run_script(self):
        if not self.image_path:
            self.output_console.setText("Please select an image file first.")
            return

        self.run_btn.setEnabled(False)
        self.run_btn.setText("Running...")
        self.output_console.setText(f"Starting analysis on {os.path.basename(self.image_path)}...\n")

        # --- KEY CHANGE: Run as Module ---
        # We use [sys.executable, "-m", "scopexr.fs_main"] instead of path to file
        command = [sys.executable, "-m"]
        
        current_tab_index = self.tab_widget.currentIndex()
        
        if current_tab_index == 0:
            # --- FOCAL SPOT ---
            command.append("scopexr.fs_main") # Module name
            command.extend(["--f", self.image_path])
            
            if self.fs_config.text():
                command.extend(["--config", self.fs_config.text()])
            if self.fs_output_dir.text():
                command.extend(["--o", self.fs_output_dir.text()])
                
            command.extend(["--p", str(self.fs_pixel_size.value())])
            command.extend(["--d", str(self.fs_diameter.value())])
            
            if self.fs_no_hough.isChecked():
                command.append("--no_hough")
            
            if self.fs_magnification.value() > 0.0:
                command.extend(["--m", str(self.fs_magnification.value())])
            
            command.extend(["--n", str(self.fs_min_pixels.value())])
            command.extend(["--nangles", str(self.fs_nangles.value())])
            command.extend(["--hl", str(self.fs_half_length.value())])
            command.extend(["--ds", str(self.fs_derivative_step.value())])
            command.extend(["--axis_shifts", str(self.fs_axis_shifts.value())])
            
            filter_text = self.fs_filter.currentText()
            if filter_text != "None":
                 command.extend(["--filter", filter_text])
            else:
                 command.extend(["--filter", "None"]) 

            command.extend(["--avg_number", str(self.fs_avg_number.value())])
            
            if self.fs_sym.isChecked():
                command.append("--sym")
            if self.fs_show.isChecked():
                command.append("--show")

            # Shifts
            if self.fs_radio_manual.isChecked():
                command.extend(["--manual_shift", str(self.fs_manual_shift_val.value())])
            elif self.fs_radio_auto.isChecked():
                command.append("--auto_shift")
            elif self.fs_radio_no.isChecked():
                command.append("--no_shift")

            # Avg
            if self.fs_avg_buttons["avg"].isChecked():
                command.append("--avg")
            elif self.fs_avg_buttons["no_avg"].isChecked():
                command.append("--no_avg")

        else:
            # --- PSF ---
            command.append("scopexr.psf_main") # Module name
            command.extend(["--f", self.image_path])

            if self.psf_config.text():
                command.extend(["--config", self.psf_config.text()])
            if self.psf_output_dir.text():
                command.extend(["--o", self.psf_output_dir.text()])
                
            command.extend(["--p", str(self.psf_pixel_size.value())])
            command.extend(["--d", str(self.psf_diameter.value())])
            
            if self.psf_no_hough.isChecked():
                command.append("--no_hough")
            
            command.extend(["--nangles", str(self.psf_nangles.value())])
            command.extend(["--hl", str(self.psf_half_length.value())])
            command.extend(["--ds", str(self.psf_derivative_step.value())])
            command.extend(["--axis_shifts", str(self.psf_axis_shifts.value())])
            
            filter_text = self.psf_filter.currentText()
            if filter_text != "None":
                 command.extend(["--filter", filter_text])
            else:
                 command.extend(["--filter", "None"])
            
            command.extend(["--avg_number", str(self.psf_avg_number.value())])

            if self.psf_sym.isChecked():
                command.append("--sym")
                
            command.extend(["--dtheta", str(self.psf_dtheta.value())])
            command.extend(["--resample1", str(self.psf_resample1.value())])
            command.extend(["--resample2", str(self.psf_resample2.value())])
            command.extend(["--gaussian_sigma", str(self.psf_gaussian_sigma.value())])
            command.extend(["--oversample_strategy", self.psf_oversample_strategy.currentText()])

            if self.psf_show.isChecked():
                command.append("--show")

            # Shifts
            if self.psf_radio_manual.isChecked():
                command.extend(["--manual_shift", str(self.psf_manual_shift_val.value())])
            elif self.psf_radio_auto.isChecked():
                command.append("--auto_shift")
            elif self.psf_radio_no.isChecked():
                command.append("--no_shift")

            # Avg
            if self.psf_avg_buttons["avg"].isChecked():
                command.append("--avg")
            elif self.psf_avg_buttons["no_avg"].isChecked():
                command.append("--no_avg")

            # Oversample
            if self.psf_oversample_buttons["oversample"].isChecked():
                command.append("--oversample")
            elif self.psf_oversample_buttons["no_oversample"].isChecked():
                command.append("--no_oversample")

        # Start Thread
        self.run_thread = RunThread(command)
        self.run_thread.output.connect(self.append_output)
        self.run_thread.finished.connect(self.on_run_finished)
        self.run_thread.start()

    def append_output(self, text):
        self.output_console.moveCursor(self.output_console.textCursor().MoveOperation.End)
        self.output_console.insertPlainText(text)
        self.output_console.ensureCursorVisible()

    def on_run_finished(self):
        self.run_btn.setEnabled(True)
        self.run_btn.setText("Run Analysis")
        self.output_console.insertPlainText("\n--- Analysis Finished ---\n")

def main():
    app = QApplication(sys.argv)
    window = ScopeXRApp()
    window.show()
    app.exec()

if __name__ == "__main__":
    main()