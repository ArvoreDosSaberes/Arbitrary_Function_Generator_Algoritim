import sys
import math
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import sympy as sp

from PySide6 import QtCore, QtWidgets


def _is_truthy_env(name: str) -> bool:
    value = (os.environ.get(name, "") or "").strip()
    return value in {"1", "true", "True", "yes", "YES"}


def _try_get_web_engine_view_class():
    if _is_truthy_env("AFG_DISABLE_WEBENGINE"):
        return None
    try:
        from PySide6.QtWebEngineWidgets import QWebEngineView  # type: ignore

        return QWebEngineView
    except Exception:
        return None

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure


DEFAULT_N_SAMPLES = 8192


@dataclass
class Waveform:
    x: np.ndarray
    y: np.ndarray


class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        fig = Figure(figsize=(7, 4), dpi=100)
        self.ax = fig.add_subplot(111)
        super().__init__(fig)
        self.setParent(parent)

        self.ax.set_title("Arbitrary Function")
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.set_xlim(0.0, 1.0)
        self.ax.set_ylim(-1.2, 1.2)
        self.ax.grid(True, alpha=0.3)

        (self._line,) = self.ax.plot([], [], lw=2)
        self._draw_points: List[Tuple[float, float]] = []
        self._drawing = False
        self._eps_x = 1e-6

        self.mpl_connect("button_press_event", self._on_press)
        self.mpl_connect("button_release_event", self._on_release)
        self.mpl_connect("motion_notify_event", self._on_move)

    def clear(self) -> None:
        self._draw_points = []
        self._line.set_data([], [])
        self.ax.set_ylim(-1.2, 1.2)
        self.draw_idle()

    def set_waveform(self, wf: Waveform) -> None:
        self._draw_points = []
        self._line.set_data(wf.x, wf.y)
        self.draw_idle()

    def set_ylim(self, ymin: float, ymax: float) -> None:
        self.ax.set_ylim(ymin, ymax)
        self.draw_idle()

    def get_drawn_points(self) -> List[Tuple[float, float]]:
        return list(self._draw_points)

    def _on_press(self, event) -> None:
        if event.inaxes != self.ax:
            return
        if event.button != 1:
            return
        self._drawing = True
        self._append_point(event.xdata, event.ydata)

    def _on_release(self, event) -> None:
        if event.button == 1:
            self._drawing = False

    def _on_move(self, event) -> None:
        if not self._drawing:
            return
        if event.inaxes != self.ax:
            return
        self._append_point(event.xdata, event.ydata)

    def _append_point(self, x: Optional[float], y: Optional[float]) -> None:
        if x is None or y is None:
            return
        x = float(np.clip(x, 0.0, 1.0))
        y = float(np.clip(y, -1.0, 1.0))
        if self._draw_points:
            last_x, _last_y = self._draw_points[-1]
            if x < (last_x - self._eps_x):
                return
            if abs(x - last_x) <= self._eps_x:
                self._draw_points[-1] = (last_x, y)
            else:
                self._draw_points.append((x, y))
        else:
            self._draw_points.append((x, y))
        xs = [p[0] for p in self._draw_points]
        ys = [p[1] for p in self._draw_points]
        self._line.set_data(xs, ys)
        self.draw_idle()


class PreviewCanvas(FigureCanvasQTAgg):
    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        fig = Figure(figsize=(3, 2), dpi=100)
        self.ax = fig.add_subplot(111)
        super().__init__(fig)
        self.setParent(parent)

        self.ax.set_xlim(0.0, 1.0)
        self.ax.set_ylim(-1.2, 1.2)
        self.ax.grid(True, alpha=0.25)
        self.ax.set_xticks([])
        self.ax.set_yticks([])

        (self._line,) = self.ax.plot([], [], lw=1.5)

    def set_waveform(self, wf: Optional[Waveform]) -> None:
        if wf is None:
            self._line.set_data([], [])
        else:
            self._line.set_data(wf.x, wf.y)
        self.draw_idle()

    def set_ylim(self, ymin: float, ymax: float) -> None:
        self.ax.set_ylim(ymin, ymax)
        self.draw_idle()


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Arbitrary Function Generator")

        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)

        self.canvas = MplCanvas(self)

        self.preview_a = PreviewCanvas(self)
        self.preview_b = PreviewCanvas(self)
        self.add_a_btn = QtWidgets.QPushButton("Adicionar A", self)
        self.add_b_btn = QtWidgets.QPushButton("Adicionar B", self)
        self.concat_a_btn = QtWidgets.QPushButton("Concatenar -> A", self)
        self.concat_b_btn = QtWidgets.QPushButton("Concatenar -> B", self)
        self.sum_btn = QtWidgets.QPushButton("Somar A + B", self)
        self.sub_btn = QtWidgets.QPushButton("Subtrair A - B", self)
        self.mul_btn = QtWidgets.QPushButton("Multiplicar A * B", self)
        self.div_btn = QtWidgets.QPushButton("Dividir A / B", self)

        self.preset_combo = QtWidgets.QComboBox(self)
        self._preset_defs = []
        self._init_presets()

        self.preset_invert_check = QtWidgets.QCheckBox("Inverter sinal (preset)", self)
        self.preset_invert_check.setChecked(False)

        self.sine_fraction_combo = QtWidgets.QComboBox(self)
        self.sine_fraction_combo.addItem("Ciclo completo (1/1)", "1")
        self.sine_fraction_combo.addItem("1/2", "1/2")
        self.sine_fraction_combo.addItem("1/3", "1/3")
        self.sine_fraction_combo.addItem("1/4", "1/4")
        self.sine_fraction_combo.addItem("1/5", "1/5")
        self.sine_fraction_combo.addItem("1/6", "1/6")
        self.sine_fraction_combo.addItem("1/7", "1/7")
        self.sine_fraction_combo.addItem("1/8", "1/8")
        self.sine_fraction_combo.addItem("1/9", "1/9")
        self.sine_fraction_combo.addItem("1/10", "1/10")

        self.triangle_opening_angle_spin = QtWidgets.QDoubleSpinBox(self)
        self.triangle_opening_angle_spin.setRange(1.0, 179.0)
        self.triangle_opening_angle_spin.setDecimals(1)
        self.triangle_opening_angle_spin.setSingleStep(1.0)
        self.triangle_opening_angle_spin.setValue(60.0)

        self.heartbeat_cycles_spin = QtWidgets.QSpinBox(self)
        self.heartbeat_cycles_spin.setRange(1, 128)
        self.heartbeat_cycles_spin.setValue(5)

        self.heartbeat_type_combo = QtWidgets.QComboBox(self)
        self.heartbeat_type_combo.addItem("Normal", "normal")
        self.heartbeat_type_combo.addItem("Taquicardia", "tachy")
        self.heartbeat_type_combo.addItem("Bradicardia", "brady")
        self.heartbeat_type_combo.addItem("PVC (extra-sístole ventricular)", "pvc")
        self.heartbeat_type_combo.addItem("Fibrilação atrial (simplificada)", "afib")

        self.bell_width_spin = QtWidgets.QDoubleSpinBox(self)
        self.bell_width_spin.setRange(0.01, 0.50)
        self.bell_width_spin.setDecimals(3)
        self.bell_width_spin.setSingleStep(0.005)
        self.bell_width_spin.setValue(0.12)

        self.bell_rise_smooth_spin = QtWidgets.QDoubleSpinBox(self)
        self.bell_rise_smooth_spin.setRange(0.20, 5.00)
        self.bell_rise_smooth_spin.setDecimals(2)
        self.bell_rise_smooth_spin.setSingleStep(0.05)
        self.bell_rise_smooth_spin.setValue(1.00)

        self.bell_fall_smooth_spin = QtWidgets.QDoubleSpinBox(self)
        self.bell_fall_smooth_spin.setRange(0.20, 5.00)
        self.bell_fall_smooth_spin.setDecimals(2)
        self.bell_fall_smooth_spin.setSingleStep(0.05)
        self.bell_fall_smooth_spin.setValue(1.00)

        self.square_duty_spin = QtWidgets.QDoubleSpinBox(self)
        self.square_duty_spin.setRange(0.1, 99.9)
        self.square_duty_spin.setDecimals(1)
        self.square_duty_spin.setSingleStep(1.0)
        self.square_duty_spin.setValue(50.0)

        self.square_bounce_rise_check = QtWidgets.QCheckBox("Recochetear na subida", self)
        self.square_bounce_rise_check.setChecked(False)
        self.square_bounce_fall_check = QtWidgets.QCheckBox("Recochetear na descida", self)
        self.square_bounce_fall_check.setChecked(False)

        self.square_bounce_amp_spin = QtWidgets.QDoubleSpinBox(self)
        self.square_bounce_amp_spin.setRange(0.0, 1.0)
        self.square_bounce_amp_spin.setDecimals(3)
        self.square_bounce_amp_spin.setSingleStep(0.02)
        self.square_bounce_amp_spin.setValue(0.15)

        self.square_bounce_duration_spin = QtWidgets.QDoubleSpinBox(self)
        self.square_bounce_duration_spin.setRange(0.0, 0.5)
        self.square_bounce_duration_spin.setDecimals(4)
        self.square_bounce_duration_spin.setSingleStep(0.005)
        self.square_bounce_duration_spin.setValue(0.03)

        self.square_rise_time_spin = QtWidgets.QDoubleSpinBox(self)
        self.square_rise_time_spin.setRange(0.0, 0.25)
        self.square_rise_time_spin.setDecimals(4)
        self.square_rise_time_spin.setSingleStep(0.002)
        self.square_rise_time_spin.setValue(0.0)

        self.square_fall_time_spin = QtWidgets.QDoubleSpinBox(self)
        self.square_fall_time_spin.setRange(0.0, 0.25)
        self.square_fall_time_spin.setDecimals(4)
        self.square_fall_time_spin.setSingleStep(0.002)
        self.square_fall_time_spin.setValue(0.0)

        self.square_bounce_decay_type_combo = QtWidgets.QComboBox(self)
        self.square_bounce_decay_type_combo.addItem("Exponencial (direto)", "exp")
        self.square_bounce_decay_type_combo.addItem("Ln (direto)", "ln")

        self.square_bounce_decay_spin = QtWidgets.QDoubleSpinBox(self)
        self.square_bounce_decay_spin.setRange(0.01, 50.0)
        self.square_bounce_decay_spin.setDecimals(3)
        self.square_bounce_decay_spin.setSingleStep(0.25)
        self.square_bounce_decay_spin.setValue(8.0)

        self.exp_log_smooth_spin = QtWidgets.QDoubleSpinBox(self)
        self.exp_log_smooth_spin.setRange(0.10, 50.0)
        self.exp_log_smooth_spin.setDecimals(2)
        self.exp_log_smooth_spin.setSingleStep(0.25)
        self.exp_log_smooth_spin.setValue(5.0)

        self.formula_edit = QtWidgets.QLineEdit(self)
        self.formula_edit.setPlaceholderText("Ex: sin(2*pi*x/256) + 0.2*sin(2*pi*x/1024)  (x = 0..N-1)")

        self.apply_formula_btn = QtWidgets.QPushButton("Aplicar fórmula", self)
        self.clear_btn = QtWidgets.QPushButton("Limpar desenho", self)
        self.save_btn = QtWidgets.QPushButton("Salvar .txt", self)
        self.load_btn = QtWidgets.QPushButton("Carregar .txt", self)

        self.sample_count_spin = QtWidgets.QSpinBox(self)
        self.sample_count_spin.setRange(16, 262144)
        self.sample_count_spin.setSingleStep(256)
        self.sample_count_spin.setValue(DEFAULT_N_SAMPLES)

        self.cycles_spin = QtWidgets.QSpinBox(self)
        self.cycles_spin.setRange(1, 64)
        self.cycles_spin.setSingleStep(1)
        self.cycles_spin.setValue(1)

        self.amp_spin = QtWidgets.QDoubleSpinBox(self)
        self.amp_spin.setRange(0.0, 10.0)
        self.amp_spin.setDecimals(4)
        self.amp_spin.setSingleStep(0.1)
        self.amp_spin.setValue(1.0)

        self.phase_spin = QtWidgets.QDoubleSpinBox(self)
        self.phase_spin.setRange(-360.0, 360.0)
        self.phase_spin.setDecimals(2)
        self.phase_spin.setSingleStep(5.0)
        self.phase_spin.setValue(0.0)

        self.offset_spin = QtWidgets.QDoubleSpinBox(self)
        self.offset_spin.setRange(-10.0, 10.0)
        self.offset_spin.setDecimals(4)
        self.offset_spin.setSingleStep(0.05)
        self.offset_spin.setValue(0.0)

        self.status = QtWidgets.QLabel("", self)
        self.status.setWordWrap(True)

        self.web_view = None
        web_engine_view_class = _try_get_web_engine_view_class()
        if web_engine_view_class is not None:
            try:
                self.web_view = web_engine_view_class(self)
                self.web_view.setMinimumHeight(120)
            except Exception:
                self.web_view = None
        if self.web_view is None:
            placeholder = QtWidgets.QLabel(
                "QtWebEngine está desabilitado/indisponível. A fórmula será plotada, mas não renderizada via MathJax.",
                self,
            )
            placeholder.setWordWrap(True)
            self.web_view = placeholder  # type: ignore

        controls = QtWidgets.QVBoxLayout()
        controls.addWidget(QtWidgets.QLabel("Presets:", self))
        controls.addWidget(self.preset_combo)
        controls.addWidget(self.preset_invert_check)

        self.sine_controls = QtWidgets.QGroupBox("Seno/Cosseno", self)
        sine_form = QtWidgets.QFormLayout(self.sine_controls)
        sine_form.addRow("Trecho do ciclo:", self.sine_fraction_combo)
        controls.addWidget(self.sine_controls)

        self.exp_log_controls = QtWidgets.QGroupBox("Exponencial/Logaritmo", self)
        exp_log_form = QtWidgets.QFormLayout(self.exp_log_controls)
        exp_log_form.addRow("Suavidade:", self.exp_log_smooth_spin)
        controls.addWidget(self.exp_log_controls)

        self.triangle_controls = QtWidgets.QGroupBox("Triangular", self)
        tri_form = QtWidgets.QFormLayout(self.triangle_controls)
        tri_form.addRow("Ângulo de abertura (graus):", self.triangle_opening_angle_spin)
        controls.addWidget(self.triangle_controls)

        self.bell_controls = QtWidgets.QGroupBox("Sino (Gaussiana)", self)
        bell_form = QtWidgets.QFormLayout(self.bell_controls)
        bell_form.addRow("Abertura (largura):", self.bell_width_spin)
        bell_form.addRow("Suavidade de subida:", self.bell_rise_smooth_spin)
        bell_form.addRow("Suavidade de descida:", self.bell_fall_smooth_spin)
        controls.addWidget(self.bell_controls)

        self.square_controls = QtWidgets.QGroupBox("Quadrada", self)
        square_form = QtWidgets.QFormLayout(self.square_controls)
        square_form.addRow("Duty Cycle (%):", self.square_duty_spin)
        square_form.addRow("Subida (fração do ciclo):", self.square_rise_time_spin)
        square_form.addRow("Descida (fração do ciclo):", self.square_fall_time_spin)
        square_form.addRow(self.square_bounce_rise_check)
        square_form.addRow(self.square_bounce_fall_check)
        square_form.addRow("Amplitude do recochete:", self.square_bounce_amp_spin)
        square_form.addRow("Duração (fração do ciclo):", self.square_bounce_duration_spin)
        square_form.addRow("Decaimento:", self.square_bounce_decay_type_combo)
        square_form.addRow("Intensidade:", self.square_bounce_decay_spin)
        controls.addWidget(self.square_controls)

        self.heartbeat_controls = QtWidgets.QGroupBox("Heartbeat", self)
        hb_form = QtWidgets.QFormLayout(self.heartbeat_controls)
        hb_form.addRow("Ciclos no buffer (bat/s):", self.heartbeat_cycles_spin)
        hb_form.addRow("Tipo:", self.heartbeat_type_combo)
        controls.addWidget(self.heartbeat_controls)

        controls.addWidget(QtWidgets.QLabel("Fórmula:", self))
        controls.addWidget(self.formula_edit)
        controls.addWidget(self.apply_formula_btn)
        controls.addSpacing(8)
        controls.addWidget(QtWidgets.QLabel("Render (MathJax):", self))
        controls.addWidget(self.web_view)  # type: ignore[arg-type]
        controls.addSpacing(8)
        controls.addWidget(QtWidgets.QLabel("Amplitude:", self))
        controls.addWidget(self.amp_spin)
        controls.addWidget(QtWidgets.QLabel("Defasagem (graus):", self))
        controls.addWidget(self.phase_spin)
        controls.addWidget(QtWidgets.QLabel("Offset:", self))
        controls.addWidget(self.offset_spin)
        controls.addSpacing(8)
        controls.addWidget(QtWidgets.QLabel("Amostras:", self))
        controls.addWidget(self.sample_count_spin)
        controls.addWidget(QtWidgets.QLabel("Ciclos:", self))
        controls.addWidget(self.cycles_spin)
        controls.addSpacing(8)
        controls.addWidget(self.load_btn)
        controls.addWidget(self.save_btn)
        controls.addStretch(1)
        controls.addWidget(self.status)

        left_panel = QtWidgets.QVBoxLayout()
        left_panel.addWidget(QtWidgets.QLabel("Slot A", self))
        left_panel.addWidget(self.preview_a)
        slot_a_btns = QtWidgets.QHBoxLayout()
        slot_a_btns.addWidget(self.add_a_btn)
        slot_a_btns.addWidget(self.concat_a_btn)
        left_panel.addLayout(slot_a_btns)
        left_panel.addSpacing(10)
        left_panel.addWidget(self.sum_btn)
        left_panel.addWidget(self.sub_btn)
        left_panel.addWidget(self.mul_btn)
        left_panel.addWidget(self.div_btn)
        left_panel.addSpacing(10)
        left_panel.addWidget(QtWidgets.QLabel("Slot B", self))
        left_panel.addWidget(self.preview_b)
        slot_b_btns = QtWidgets.QHBoxLayout()
        slot_b_btns.addWidget(self.add_b_btn)
        slot_b_btns.addWidget(self.concat_b_btn)
        left_panel.addLayout(slot_b_btns)
        left_panel.addStretch(1)

        center_panel = QtWidgets.QVBoxLayout()
        center_panel.addWidget(self.canvas)
        center_panel.addWidget(self.clear_btn)

        main = QtWidgets.QHBoxLayout(central)
        main.addLayout(left_panel, stretch=1)
        main.addLayout(center_panel, stretch=3)
        main.addLayout(controls, stretch=2)

        self.apply_formula_btn.clicked.connect(self._on_apply_formula)
        self.clear_btn.clicked.connect(self._on_clear)
        self.load_btn.clicked.connect(self._on_load)
        self.save_btn.clicked.connect(self._on_save)
        self.formula_edit.textChanged.connect(self._on_formula_changed)
        self.preset_combo.currentIndexChanged.connect(self._on_preset_changed)
        self.preset_invert_check.stateChanged.connect(self._on_preset_invert_changed)
        self.sine_fraction_combo.currentIndexChanged.connect(self._on_sine_fraction_changed)
        self.triangle_opening_angle_spin.valueChanged.connect(self._on_preset_params_changed)
        self.bell_width_spin.valueChanged.connect(self._on_preset_params_changed)
        self.bell_rise_smooth_spin.valueChanged.connect(self._on_preset_params_changed)
        self.bell_fall_smooth_spin.valueChanged.connect(self._on_preset_params_changed)
        self.square_duty_spin.valueChanged.connect(self._on_preset_params_changed)
        self.square_rise_time_spin.valueChanged.connect(self._on_preset_params_changed)
        self.square_fall_time_spin.valueChanged.connect(self._on_preset_params_changed)
        self.square_bounce_rise_check.stateChanged.connect(self._on_preset_params_changed)
        self.square_bounce_fall_check.stateChanged.connect(self._on_preset_params_changed)
        self.square_bounce_amp_spin.valueChanged.connect(self._on_preset_params_changed)
        self.square_bounce_duration_spin.valueChanged.connect(self._on_preset_params_changed)
        self.square_bounce_decay_type_combo.currentIndexChanged.connect(self._on_preset_params_changed)
        self.square_bounce_decay_spin.valueChanged.connect(self._on_preset_params_changed)
        self.exp_log_smooth_spin.valueChanged.connect(self._on_exp_log_smooth_changed)
        self.heartbeat_cycles_spin.valueChanged.connect(self._on_heartbeat_params_changed)
        self.heartbeat_type_combo.currentIndexChanged.connect(self._on_heartbeat_params_changed)
        self.sample_count_spin.valueChanged.connect(self._on_sample_count_changed)
        self.cycles_spin.valueChanged.connect(self._on_transform_changed)
        self.amp_spin.valueChanged.connect(self._on_transform_changed)
        self.phase_spin.valueChanged.connect(self._on_transform_changed)
        self.offset_spin.valueChanged.connect(self._on_transform_changed)

        self.add_a_btn.clicked.connect(self._on_add_a)
        self.add_b_btn.clicked.connect(self._on_add_b)
        self.sum_btn.clicked.connect(self._on_sum)
        self.sub_btn.clicked.connect(self._on_sub)
        self.mul_btn.clicked.connect(self._on_mul)
        self.div_btn.clicked.connect(self._on_div)
        self.concat_a_btn.clicked.connect(self._on_concat_a)
        self.concat_b_btn.clicked.connect(self._on_concat_b)

        self._current_waveform_base: Optional[Waveform] = None
        self._slot_a_base: Optional[Waveform] = None
        self._slot_b_base: Optional[Waveform] = None
        self._slot_a_cycles: List[Waveform] = []
        self._slot_b_cycles: List[Waveform] = []
        self._suppress_custom_switch = False
        self.preview_a.set_waveform(None)
        self.preview_b.set_waveform(None)
        self._on_formula_changed(self.formula_edit.text())
        self._update_heartbeat_controls()
        self._update_preset_invert_control()
        self._update_sine_controls()
        self._update_exp_log_controls()
        self._update_triangle_controls()
        self._update_bell_controls()
        self._update_square_controls()

        self._update_save_label()

    def _init_presets(self) -> None:
        # kind: 'expr' usa SymPy; kind: 'gen' gera diretamente 8192 amostras
        self.preset_combo.clear()
        self.preset_combo.addItem("Custom", {"kind": "custom"})

        self.preset_combo.addItem("AbsSine", {"kind": "gen", "gen": "abs_sine"})
        self.preset_combo.addItem("AmpALT", {"kind": "gen", "gen": "amp_alt"})
        self.preset_combo.addItem("AttALT", {"kind": "gen", "gen": "att_alt"})

        self.preset_combo.addItem("Seno (sin)", {"kind": "expr", "expr_id": "sin", "expr": "sin(2*pi*x/8191)"})
        self.preset_combo.addItem("Cosseno (cos)", {"kind": "expr", "expr_id": "cos", "expr": "cos(2*pi*x/8191)"})
        self.preset_combo.addItem("CosH", {"kind": "gen", "gen": "cosh"})
        self.preset_combo.addItem("Exponencial (exp)", {"kind": "expr", "expr_id": "exp", "expr": "(exp(5*(x/8191)) - 1) / (exp(5) - 1)"})
        self.preset_combo.addItem(
            "Logaritmo natural (ln)",
            {"kind": "expr", "expr_id": "log", "expr": "log(1 + 9*(x/8191)) / log(10)"},
        )

        self.preset_combo.addItem("LogNormal", {"kind": "gen", "gen": "lognormal"})
        self.preset_combo.addItem("Log2_up", {"kind": "gen", "gen": "log2_up"})
        self.preset_combo.addItem("Log2_down", {"kind": "gen", "gen": "log2_down"})

        self.preset_combo.addItem("Quadrada", {"kind": "gen", "gen": "square"})
        self.preset_combo.addItem("Dente de serra", {"kind": "expr", "expr": "x/8191"})
        self.preset_combo.addItem("Rampa", {"kind": "expr", "expr": "x/8191"})
        self.preset_combo.addItem("Triangular", {"kind": "gen", "gen": "triangular"})
        self.preset_combo.addItem("tri_up", {"kind": "gen", "gen": "tri_up"})
        self.preset_combo.addItem("tri_down", {"kind": "gen", "gen": "tri_down"})
        self.preset_combo.addItem("Trapezia", {"kind": "gen", "gen": "trapezia"})
        self.preset_combo.addItem("StairUD", {"kind": "gen", "gen": "stair_ud"})
        self.preset_combo.addItem("StepResp", {"kind": "gen", "gen": "step_resp"})

        self.preset_combo.addItem("Sinc", {"kind": "gen", "gen": "sinc"})
        self.preset_combo.addItem("Lorentz", {"kind": "gen", "gen": "lorenz"})

        self.preset_combo.addItem("GaussianMonopulse", {"kind": "gen", "gen": "gaussian_monopulse"})
        self.preset_combo.addItem("GaussPulse", {"kind": "gen", "gen": "gauss_pulse"})
        self.preset_combo.addItem("Radar", {"kind": "gen", "gen": "radar"})

        self.preset_combo.addItem("Sino comum (Gaussiana)", {"kind": "gen", "gen": "gaussian_bell"})
        self.preset_combo.addItem("Sino invertido", {"kind": "gen", "gen": "gaussian_bell_inverted"})

        self.preset_combo.addItem("Ruído branco", {"kind": "gen", "gen": "white_noise"})
        self.preset_combo.addItem("Ruído rosa", {"kind": "gen", "gen": "pink_noise"})
        self.preset_combo.addItem("Cardiac", {"kind": "gen", "gen": "heartbeat"})
        self.preset_combo.addItem("Pulseogram", {"kind": "gen", "gen": "pulseogram"})
        self.preset_combo.addItem("EEG", {"kind": "gen", "gen": "eeg"})
        self.preset_combo.addItem("EOG", {"kind": "gen", "gen": "eog"})
        self.preset_combo.addItem("TV", {"kind": "gen", "gen": "tv"})
        self.preset_combo.addItem("VOICE", {"kind": "gen", "gen": "voice"})
        self.preset_combo.addItem("SineVer", {"kind": "gen", "gen": "sine_ver"})

    def _update_heartbeat_controls(self) -> None:
        data = self.preset_combo.currentData()
        enabled = bool(isinstance(data, dict) and data.get("kind") == "gen" and data.get("gen") == "heartbeat")
        self.heartbeat_controls.setVisible(enabled)

    def _update_preset_invert_control(self) -> None:
        data = self.preset_combo.currentData()
        enabled = bool(isinstance(data, dict) and data.get("kind") in {"expr", "gen"})
        self.preset_invert_check.setEnabled(enabled)

    def _update_sine_controls(self) -> None:
        data = self.preset_combo.currentData()
        enabled = bool(
            isinstance(data, dict)
            and data.get("kind") == "expr"
            and data.get("expr_id") in {"sin", "cos"}
        )
        self.sine_controls.setVisible(enabled)

    def _update_exp_log_controls(self) -> None:
        data = self.preset_combo.currentData()
        enabled = bool(
            isinstance(data, dict)
            and data.get("kind") == "expr"
            and data.get("expr_id") in {"exp", "log"}
        )
        self.exp_log_controls.setVisible(enabled)

    def _update_triangle_controls(self) -> None:
        data = self.preset_combo.currentData()
        enabled = bool(isinstance(data, dict) and data.get("kind") == "gen" and data.get("gen") == "triangular")
        self.triangle_controls.setVisible(enabled)

    def _update_bell_controls(self) -> None:
        data = self.preset_combo.currentData()
        enabled = bool(
            isinstance(data, dict)
            and data.get("kind") == "gen"
            and data.get("gen") in {"gaussian_bell", "gaussian_bell_inverted"}
        )
        self.bell_controls.setVisible(enabled)

    def _update_square_controls(self) -> None:
        data = self.preset_combo.currentData()
        enabled = bool(isinstance(data, dict) and data.get("kind") == "gen" and data.get("gen") == "square")
        self.square_controls.setVisible(enabled)

    def _on_sine_fraction_changed(self, _index: int) -> None:
        # Atualiza o campo de fórmula somente se o preset atual for seno/cosseno.
        data = self.preset_combo.currentData()
        if not (isinstance(data, dict) and data.get("kind") == "expr" and data.get("expr_id") in {"sin", "cos"}):
            return
        expr_id = str(data.get("expr_id"))
        fraction = str(self.sine_fraction_combo.currentData() or "1")
        denom = self._get_sample_denom()
        self._set_formula_text_programmatically(f"{expr_id}(2*pi*(x/{denom})*({fraction}))")

    def _on_exp_log_smooth_changed(self) -> None:
        data = self.preset_combo.currentData()
        if not (isinstance(data, dict) and data.get("kind") == "expr" and data.get("expr_id") in {"exp", "log"}):
            return
        self._update_exp_log_formula()

    def _update_exp_log_formula(self) -> None:
        data = self.preset_combo.currentData()
        if not (isinstance(data, dict) and data.get("kind") == "expr" and data.get("expr_id") in {"exp", "log"}):
            return
        expr_id = str(data.get("expr_id"))
        k = float(self.exp_log_smooth_spin.value())
        denom = self._get_sample_denom()

        if expr_id == "exp":
            self._set_formula_text_programmatically(f"(exp({k}*(x/{denom})) - 1) / (exp({k}) - 1)")
        else:
            self._set_formula_text_programmatically(f"log(1 + {k}*(x/{denom})) / log(1 + {k})")

    def _on_preset_invert_changed(self) -> None:
        # Inversão é aplicada ao clicar em Aplicar.
        pass

    def _on_preset_params_changed(self) -> None:
        # Mantém UI responsiva; geração só ocorre ao clicar em Aplicar.
        pass

    def _on_heartbeat_params_changed(self) -> None:
        # Mantém UI responsiva; geração só ocorre ao clicar em Aplicar.
        pass

    def _on_preset_changed(self, _index: int) -> None:
        data = self.preset_combo.currentData()
        if not isinstance(data, dict):
            return
        if data.get("kind") == "expr":
            expr = str(data.get("expr", ""))
            if expr:
                self._set_formula_text_programmatically(expr)
        self._update_heartbeat_controls()
        self._update_preset_invert_control()
        self._update_sine_controls()
        self._update_exp_log_controls()
        self._update_triangle_controls()
        self._update_bell_controls()
        self._update_square_controls()

        if data.get("kind") == "expr" and data.get("expr_id") in {"sin", "cos"}:
            self._on_sine_fraction_changed(self.sine_fraction_combo.currentIndex())
        if data.get("kind") == "expr" and data.get("expr_id") in {"exp", "log"}:
            self._update_exp_log_formula()

    def _gauss(self, t: np.ndarray, mu: float, sigma: float) -> np.ndarray:
        return np.exp(-0.5 * ((t - mu) / sigma) ** 2)

    def _get_sample_count(self) -> int:
        return int(self.sample_count_spin.value())

    def _get_sample_denom(self) -> int:
        # Evita divisão por zero em expressões do tipo x/(N-1)
        return max(1, self._get_sample_count() - 1)

    def _get_cycles(self) -> int:
        return int(max(1, self.cycles_spin.value()))

    def _update_save_label(self) -> None:
        n = self._get_sample_count()
        self.save_btn.setText(f"Salvar .txt ({n} amostras)")

    def _resample_waveform(self, wf: Waveform, n_new: int) -> Waveform:
        grid_new = np.linspace(0.0, 1.0, n_new, endpoint=False)
        y_new = np.interp(grid_new, wf.x, wf.y)
        return Waveform(x=grid_new, y=y_new)

    def _compose_equal_cycles(self, cycles: List[Waveform], n_target: int) -> Optional[Waveform]:
        if not cycles:
            return None
        m = int(len(cycles))
        n_target = int(max(2, n_target))

        base_len = n_target // m
        rem = n_target - (base_len * m)
        if base_len <= 0:
            base_len = 1
            rem = 0

        y_out = np.zeros(n_target, dtype=float)
        idx = 0
        for i, wf in enumerate(cycles):
            seg_len = base_len + (1 if i < rem else 0)
            if seg_len <= 0:
                continue
            wf_seg = self._resample_waveform(wf, seg_len) if wf.y.size != seg_len else wf
            y_out[idx : idx + seg_len] = wf_seg.y[:seg_len]
            idx += seg_len

        grid = np.linspace(0.0, 1.0, n_target, endpoint=False)
        return Waveform(x=grid, y=y_out)

    def _on_sample_count_changed(self) -> None:
        n = self._get_sample_count()

        if self._current_waveform_base is not None and self._current_waveform_base.y.size != n:
            self._current_waveform_base = self._resample_waveform(self._current_waveform_base, n)

        if self._slot_a_cycles:
            self._slot_a_cycles = [self._resample_waveform(w, n) for w in self._slot_a_cycles]
            self._slot_a_base = self._compose_equal_cycles(self._slot_a_cycles, n)
        elif self._slot_a_base is not None and self._slot_a_base.y.size != n:
            self._slot_a_base = self._resample_waveform(self._slot_a_base, n)

        if self._slot_b_cycles:
            self._slot_b_cycles = [self._resample_waveform(w, n) for w in self._slot_b_cycles]
            self._slot_b_base = self._compose_equal_cycles(self._slot_b_cycles, n)
        elif self._slot_b_base is not None and self._slot_b_base.y.size != n:
            self._slot_b_base = self._resample_waveform(self._slot_b_base, n)

        self._update_save_label()
        self._refresh_previews()
        if self._current_waveform_base is not None:
            self._refresh_main_plot()

    def _set_formula_text_programmatically(self, text: str) -> None:
        self._suppress_custom_switch = True
        try:
            self.formula_edit.setText(text)
        finally:
            self._suppress_custom_switch = False

    def _heartbeat_template(self, n: int, kind: str) -> np.ndarray:
        # Template simples de um ciclo PQRST, baseado em literatura de simulação por morfologia
        # (ex.: abordagem por picos gaussianos e/ou ecgsyn). Aqui é propositalmente simplificado.
        t = np.linspace(0.0, 1.0, n, endpoint=False)

        # (mu, sigma, amp)
        P = (0.18, 0.03, 0.12)
        Q = (0.36, 0.010, -0.18)
        R = (0.40, 0.012, 1.00)
        S = (0.43, 0.012, -0.35)
        T = (0.62, 0.050, 0.30)

        if kind == "tachy":
            # QRS levemente mais estreito e T um pouco menor
            Q = (Q[0], Q[1] * 0.85, Q[2])
            R = (R[0], R[1] * 0.85, R[2])
            S = (S[0], S[1] * 0.85, S[2])
            T = (T[0], T[1], T[2] * 0.85)
        elif kind == "brady":
            # T um pouco mais largo
            T = (T[0], T[1] * 1.15, T[2])
        elif kind == "pvc":
            # PVC simplificado: sem onda P, QRS mais largo e T invertida
            P = (P[0], P[1], 0.0)
            Q = (0.35, 0.020, -0.25)
            R = (0.40, 0.030, 1.10)
            S = (0.46, 0.025, -0.55)
            T = (0.70, 0.070, -0.25)
        elif kind == "afib":
            # AFib simplificado: remove P e adiciona baseline fibrilatória
            P = (P[0], P[1], 0.0)

        y = (
            P[2] * self._gauss(t, P[0], P[1])
            + Q[2] * self._gauss(t, Q[0], Q[1])
            + R[2] * self._gauss(t, R[0], R[1])
            + S[2] * self._gauss(t, S[0], S[1])
            + T[2] * self._gauss(t, T[0], T[1])
        )

        if kind == "afib":
            y = y + 0.03 * np.sin(2 * np.pi * (6.0 * t)) + 0.02 * np.sin(2 * np.pi * (9.0 * t + 0.2))

        # Normaliza para [-1, 1]
        y = y - np.mean(y)
        y = y / (np.max(np.abs(y)) + 1e-12)
        return y

    def _generate_heartbeat(self, cycles: int, kind: str, n_samples: int) -> np.ndarray:
        # Interpreta o buffer como 1 segundo: cycles = batimentos por segundo no buffer.
        cycles = int(max(1, cycles))

        if kind == "tachy":
            cycles = max(cycles, 8)
        elif kind == "brady":
            cycles = min(cycles, 2)

        # Gera intervalos RR em amostras (soma = n_samples)
        if kind == "afib":
            rr = np.random.uniform(0.6, 1.4, size=cycles)
            rr = rr / rr.sum() * n_samples
            rr = np.maximum(16.0, rr)
            rr = rr / rr.sum() * n_samples
            rr_int = np.floor(rr).astype(int)
            rr_int[-1] += n_samples - int(rr_int.sum())
        elif kind == "pvc":
            # Uma extra-sístole a cada 4 batimentos: curto + compensatório.
            base = n_samples / cycles
            rr_int = np.full(cycles, int(round(base)), dtype=int)
            if cycles >= 4:
                i = cycles // 2
                short = max(16, int(round(base * 0.65)))
                long = max(16, int(round(base * 1.35)))
                rr_int[i] = short
                rr_int[min(i + 1, cycles - 1)] = long
            rr_int[-1] += n_samples - int(rr_int.sum())
        else:
            rr_int = np.full(cycles, n_samples // cycles, dtype=int)
            rr_int[-1] += n_samples - int(rr_int.sum())

        y = np.zeros(n_samples, dtype=float)
        idx = 0
        for b in range(cycles):
            n = int(rr_int[b])
            if n <= 0:
                continue
            beat_kind = kind
            if kind == "pvc" and cycles >= 4 and b == cycles // 2:
                beat_kind = "pvc"
            tmpl = self._heartbeat_template(n, beat_kind)
            y[idx : idx + n] = tmpl[:n]
            idx += n

        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        y = y / (np.max(np.abs(y)) + 1e-12) * 0.9
        return y

    def _generate_preset(self, gen_name: str) -> Waveform:
        n = self._get_sample_count()
        grid = np.linspace(0.0, 1.0, n, endpoint=False)
        x = np.arange(n, dtype=float)
        do_clip = True

        if gen_name == "white_noise":
            y = np.random.normal(0.0, 0.35, size=n)
        elif gen_name == "abs_sine":
            y = np.abs(np.sin(2.0 * np.pi * grid))
            y = y / (np.max(np.abs(y)) + 1e-12) * 0.9
        elif gen_name == "amp_alt":
            s = np.sin(2.0 * np.pi * grid)
            a = np.where(grid < 0.5, 1.0, 0.35)
            y = s * a
            y = y / (np.max(np.abs(y)) + 1e-12) * 0.9
        elif gen_name == "att_alt":
            s = np.sin(2.0 * np.pi * grid)
            k = 6.0
            env1 = np.exp(-k * (grid / 0.5))
            env2 = np.exp(-k * ((grid - 0.5) / 0.5))
            env = np.where(grid < 0.5, env1, env2)
            y = s * env
            y = y / (np.max(np.abs(y)) + 1e-12) * 0.9
        elif gen_name == "square":
            do_clip = False
            duty = float(self.square_duty_spin.value()) / 100.0
            duty = float(np.clip(duty, 1e-6, 1.0 - 1e-6))

            rise_w = float(self.square_rise_time_spin.value())
            fall_w = float(self.square_fall_time_spin.value())
            rise_w = float(np.clip(rise_w, 0.0, 0.25))
            fall_w = float(np.clip(fall_w, 0.0, 0.25))

            bounce_amp = float(self.square_bounce_amp_spin.value())
            bounce_duration = float(self.square_bounce_duration_spin.value())
            bounce_duration = float(np.clip(bounce_duration, 0.0, 0.5))
            decay_k = float(self.square_bounce_decay_spin.value())
            decay_kind = str(self.square_bounce_decay_type_combo.currentData() or "exp")

            rise_on = bool(self.square_bounce_rise_check.isChecked())
            fall_on = bool(self.square_bounce_fall_check.isChecked())

            t = grid

            def sigmoid(z: np.ndarray) -> np.ndarray:
                return 0.5 * (1.0 + np.tanh(z))

            def step(t0: np.ndarray, edge: float) -> np.ndarray:
                return (t0 >= edge).astype(float)

            def wrapped_dist(t0: np.ndarray, center: float) -> np.ndarray:
                # intervalo [-0.5, 0.5)
                return ((t0 - center + 0.5) % 1.0) - 0.5

            # Transição de subida em t=0 (circular) e de descida em t=duty.
            if rise_w <= 0.0:
                r = step(t, 0.0)
            else:
                r = sigmoid(wrapped_dist(t, 0.0) / max(1e-12, rise_w))

            if fall_w <= 0.0:
                f = step(t, duty)
            else:
                f = sigmoid((t - duty) / max(1e-12, fall_w))

            base = r * (1.0 - f)
            y = base.copy()

            if bounce_amp > 0.0 and bounce_duration > 0.0 and (rise_on or fall_on):
                def decay(u: np.ndarray) -> np.ndarray:
                    u = np.clip(u, 0.0, 1.0)
                    if decay_kind == "exp":
                        return np.exp(-decay_k * u)
                    if decay_kind == "ln":
                        return 1.0 - (np.log1p(decay_k * u) / np.log1p(decay_k))
                    return np.exp(-decay_k * u)

                # Oscilação fixa dentro da janela: 6 ciclos na duração escolhida
                edge_freq = 6.0

                def add_bounce(edge_t: float, direction: float) -> None:
                    dt = (t - edge_t) % 1.0
                    mask = dt < bounce_duration
                    if not np.any(mask):
                        return
                    u = dt[mask] / max(1e-12, bounce_duration)
                    # cos() inicia em 1.0: gera overshoot imediato (subida acima, descida abaixo)
                    ring = np.cos(2.0 * np.pi * edge_freq * u) * decay(u)
                    y[mask] = base[mask] + direction * bounce_amp * ring

                # Subida em t=0 (início do ciclo)
                if rise_on:
                    add_bounce(edge_t=0.0, direction=+1.0)
                # Descida em t=duty
                if fall_on:
                    add_bounce(edge_t=duty, direction=-1.0)

            y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        elif gen_name == "tri_up":
            y = grid * 2.0 - 1.0
            y = np.clip(y, -1.0, 1.0)
        elif gen_name == "tri_down":
            y = 1.0 - (grid * 2.0)
            y = np.clip(y, -1.0, 1.0)
        elif gen_name == "trapezia":
            rise = 0.15
            high = 0.35
            fall = 0.15
            low = 1.0 - (rise + high + fall)
            t = grid
            y = np.empty_like(t)
            a0 = rise
            a1 = rise + high
            a2 = rise + high + fall
            y[t < a0] = (t[t < a0] / max(1e-12, rise))
            y[(t >= a0) & (t < a1)] = 1.0
            y[(t >= a1) & (t < a2)] = 1.0 - ((t[(t >= a1) & (t < a2)] - a1) / max(1e-12, fall))
            y[t >= a2] = 0.0
            y = y * 2.0 - 1.0
            y = np.clip(y, -1.0, 1.0)
        elif gen_name == "stair_ud":
            steps = 8
            half = n // 2
            y = np.zeros(n, dtype=float)
            for i in range(half):
                y[i] = np.floor((i / max(1, half - 1)) * steps) / max(1, steps - 1)
            for i in range(half, n):
                j = i - half
                y[i] = 1.0 - (np.floor((j / max(1, n - half - 1)) * steps) / max(1, steps - 1))
            y = y * 2.0 - 1.0
            y = np.clip(y, -1.0, 1.0)
        elif gen_name == "step_resp":
            k = 10.0
            t = grid
            y = 1.0 - np.exp(-k * t)
            y = y * 2.0 - 1.0
            y = y / (np.max(np.abs(y)) + 1e-12) * 0.9
        elif gen_name == "cosh":
            t = (grid - 0.5) * 6.0
            y = np.cosh(t)
            y = y - np.mean(y)
            y = y / (np.max(np.abs(y)) + 1e-12) * 0.9
        elif gen_name == "gauss_pulse":
            t = grid
            sigma = 0.06
            y = np.exp(-0.5 * ((t - 0.5) / max(1e-6, sigma)) ** 2)
            y = y - np.mean(y)
            y = y / (np.max(np.abs(y)) + 1e-12) * 0.9
        elif gen_name == "gaussian_monopulse":
            t = grid
            sigma = 0.06
            z = (t - 0.5) / max(1e-6, sigma)
            y = -z * np.exp(-0.5 * (z**2))
            y = y / (np.max(np.abs(y)) + 1e-12) * 0.9
        elif gen_name == "lognormal":
            eps = 1e-6
            u = np.clip(grid, eps, 1.0 - eps)
            mu = -1.0
            sigma = 0.35
            y = (1.0 / (u * sigma * np.sqrt(2.0 * np.pi))) * np.exp(-((np.log(u) - mu) ** 2) / (2.0 * sigma * sigma))
            y = y - np.mean(y)
            y = y / (np.max(np.abs(y)) + 1e-12) * 0.9
        elif gen_name == "log2_up":
            k = 6.0
            t = grid
            y = np.log2(1.0 + (2.0**k - 1.0) * t) / k
            y = y * 2.0 - 1.0
            y = y / (np.max(np.abs(y)) + 1e-12) * 0.9
        elif gen_name == "log2_down":
            k = 6.0
            t = grid
            y = 1.0 - (np.log2(1.0 + (2.0**k - 1.0) * t) / k)
            y = y * 2.0 - 1.0
            y = y / (np.max(np.abs(y)) + 1e-12) * 0.9
        elif gen_name == "pulseogram":
            t = grid
            y = np.zeros_like(t)
            centers = np.array([0.12, 0.34, 0.58, 0.82])
            widths = np.array([0.015, 0.03, 0.02, 0.025])
            amps = np.array([1.0, 0.6, 0.85, 0.7])
            for c, w, a in zip(centers, widths, amps):
                y += a * np.exp(-0.5 * ((t - c) / max(1e-6, w)) ** 2)
            y = y - np.mean(y)
            y = y / (np.max(np.abs(y)) + 1e-12) * 0.9
        elif gen_name == "radar":
            t = grid
            f0 = 3.0
            f1 = 45.0
            k = f1 - f0
            phase = 2.0 * np.pi * (f0 * t + 0.5 * k * (t**2))
            env = np.exp(-0.5 * ((t - 0.5) / 0.25) ** 2)
            y = np.sin(phase) * env
            y = y / (np.max(np.abs(y)) + 1e-12) * 0.9
        elif gen_name == "eeg":
            t = grid
            rng = np.random.default_rng()
            y = np.zeros_like(t)
            for f, a in [(2.0, 0.25), (6.0, 0.35), (10.0, 0.45), (18.0, 0.25), (30.0, 0.15)]:
                phi = float(rng.uniform(0.0, 2.0 * np.pi))
                y += a * np.sin(2.0 * np.pi * f * t + phi)
            y += 0.08 * rng.normal(0.0, 1.0, size=n)
            y = y / (np.max(np.abs(y)) + 1e-12) * 0.9
        elif gen_name == "eog":
            t = grid
            y = 0.25 * np.sin(2.0 * np.pi * 0.6 * t) + 0.18 * np.sin(2.0 * np.pi * 1.2 * t + 1.0)
            centers = np.array([0.20, 0.55, 0.78])
            for c in centers:
                y += 0.55 * np.exp(-0.5 * ((t - c) / 0.01) ** 2)
            y = y - np.mean(y)
            y = y / (np.max(np.abs(y)) + 1e-12) * 0.9
        elif gen_name == "voice":
            t = grid
            f0 = 7.0
            y = np.zeros_like(t)
            for h in range(1, 10):
                y += (1.0 / h) * np.sin(2.0 * np.pi * (f0 * h) * t)
            env = 0.5 * (1.0 - np.cos(2.0 * np.pi * t))
            y = y * env
            y = y / (np.max(np.abs(y)) + 1e-12) * 0.9
        elif gen_name == "tv":
            t = grid
            y = np.full_like(t, 0.3)
            sync_w = 0.06
            burst_start = 0.10
            burst_w = 0.10
            y[t < sync_w] = -1.0
            mask = (t >= burst_start) & (t < burst_start + burst_w)
            y[mask] = 0.0 + 0.25 * np.sin(2.0 * np.pi * 40.0 * (t[mask] - burst_start))
            y = y - np.mean(y)
            y = y / (np.max(np.abs(y)) + 1e-12) * 0.9
        elif gen_name == "sine_ver":
            t = grid
            f0 = 1.0
            f1 = 12.0
            y = np.sin(2.0 * np.pi * (f0 + (f1 - f0) * t) * t)
            y = y / (np.max(np.abs(y)) + 1e-12) * 0.9
        elif gen_name in {"gaussian_bell", "gaussian_bell_inverted"}:
            t = grid
            mu = 0.5

            base_sigma = float(self.bell_width_spin.value())
            rise_mult = float(self.bell_rise_smooth_spin.value())
            fall_mult = float(self.bell_fall_smooth_spin.value())

            sigma_left = max(1e-6, base_sigma * rise_mult)
            sigma_right = max(1e-6, base_sigma * fall_mult)

            y = np.empty_like(t)
            left = t < mu
            y[left] = np.exp(-0.5 * ((t[left] - mu) / sigma_left) ** 2)
            y[~left] = np.exp(-0.5 * ((t[~left] - mu) / sigma_right) ** 2)

            y = y / (np.max(np.abs(y)) + 1e-12) * 0.9
            if gen_name == "gaussian_bell_inverted":
                y = -y
        elif gen_name == "triangular":
            angle_deg = float(self.triangle_opening_angle_spin.value())
            angle_deg = min(max(angle_deg, 1.0), 179.0)

            half_angle_rad = np.deg2rad(angle_deg / 2.0)
            width = 2.0 * np.tan(half_angle_rad)
            width = float(np.clip(width, 1e-6, 1.0))

            t = grid
            y = 1.0 - (np.abs(t - 0.5) / (width / 2.0))
            y = np.clip(y, 0.0, 1.0)
            y = y * 0.9
        elif gen_name == "sinc":
            # sinc normalizada: sin(pi*t)/(pi*t), com t centrado no meio do buffer
            t = (x - (n - 1) / 2.0) / 512.0
            y = np.ones_like(t)
            mask = np.abs(t) > 1e-12
            y[mask] = np.sin(np.pi * t[mask]) / (np.pi * t[mask])
            y = y - np.mean(y)
            y = y / (np.max(np.abs(y)) + 1e-12) * 0.9
        elif gen_name == "lorenz":
            # Sistema de Lorenz (σ=10, ρ=28, β=8/3). Integração RK4 simples.
            omega0 = 0.0
            gamma = 0.18
            a = 1.0

            omega = (grid - 0.5) * 2.0
            half_gamma = gamma / 2.0
            y = a * (half_gamma) / (((omega - omega0) ** 2) + (half_gamma**2))
            y = y - np.min(y)
            y = y / (np.max(np.abs(y)) + 1e-12) * 0.9
        elif gen_name == "pink_noise":
            y = np.random.normal(0.0, 1.0, size=n)
            Y = np.fft.rfft(y)
            f = np.fft.rfftfreq(n)
            scale = np.ones_like(f)
            scale[1:] = 1.0 / np.sqrt(f[1:])
            Y *= scale
            y = np.fft.irfft(Y, n=n)
            y = y / (np.max(np.abs(y)) + 1e-12) * 0.9
        elif gen_name == "heartbeat":
            cycles = int(self.heartbeat_cycles_spin.value())
            kind = str(self.heartbeat_type_combo.currentData() or "normal")
            y = self._generate_heartbeat(cycles=cycles, kind=kind, n_samples=n)
        else:
            raise ValueError(f"Preset gerador desconhecido: {gen_name}")

        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        if do_clip:
            y = np.clip(y, -1.0, 1.0)
        return Waveform(x=grid, y=y)

    def _set_status(self, msg: str) -> None:
        self.status.setText(msg)

    def _apply_transform(self, wf: Waveform, *, clip: bool) -> Waveform:
        y = wf.y

        cycles = self._get_cycles()
        if cycles > 1 and y.size >= 2:
            y_rep = np.tile(y, cycles)
            grid_rep = np.linspace(0.0, 1.0, int(y_rep.size), endpoint=False)
            grid_new = np.linspace(0.0, 1.0, int(y.size), endpoint=False)
            y = np.interp(grid_new, grid_rep, y_rep)

        phase_deg = float(self.phase_spin.value())
        if phase_deg != 0.0:
            n = int(wf.y.size)
            shift = int(round((phase_deg / 360.0) * n))
            y = np.roll(y, shift)

        amp = float(self.amp_spin.value())
        if amp != 1.0:
            y = y * amp

        offset = float(self.offset_spin.value())
        if offset != 0.0:
            y = y + offset

        if clip:
            y = np.clip(y, -1.0, 1.0)
        return Waveform(x=wf.x, y=y)

    def _autoscale_y(self, y: np.ndarray) -> Tuple[float, float]:
        if y.size == 0:
            return (-1.2, 1.2)
        y_min = float(np.nanmin(y))
        y_max = float(np.nanmax(y))
        if not np.isfinite(y_min) or not np.isfinite(y_max):
            return (-1.2, 1.2)
        if y_min == y_max:
            span = 1.0
            y_min -= span * 0.5
            y_max += span * 0.5
        pad = 0.08 * (y_max - y_min)
        pad = max(pad, 0.2)
        return (y_min - pad, y_max + pad)

    def _refresh_main_plot(self) -> None:
        wf_base = self._current_waveform_base
        if wf_base is None:
            return
        wf_disp = self._apply_transform(wf_base, clip=False)
        self.canvas.set_waveform(wf_disp)
        ymin, ymax = self._autoscale_y(wf_disp.y)
        self.canvas.set_ylim(ymin, ymax)

    def _refresh_previews(self) -> None:
        if self._slot_a_base is not None:
            wf_a = self._apply_transform(self._slot_a_base, clip=False)
            self.preview_a.set_waveform(wf_a)
            ymin, ymax = self._autoscale_y(wf_a.y)
            self.preview_a.set_ylim(ymin, ymax)
        else:
            self.preview_a.set_waveform(None)
            self.preview_a.set_ylim(-1.2, 1.2)

        if self._slot_b_base is not None:
            wf_b = self._apply_transform(self._slot_b_base, clip=False)
            self.preview_b.set_waveform(wf_b)
            ymin, ymax = self._autoscale_y(wf_b.y)
            self.preview_b.set_ylim(ymin, ymax)
        else:
            self.preview_b.set_waveform(None)
            self.preview_b.set_ylim(-1.2, 1.2)

    def _on_transform_changed(self) -> None:
        # Atualiza plot principal e previews ao mudar amplitude/defasagem/offset
        if self._current_waveform_base is not None:
            self._refresh_main_plot()
        self._refresh_previews()

    def _on_clear(self) -> None:
        self._current_waveform_base = None
        self.canvas.clear()
        self._set_status("Desenho limpo. Desenhe com o mouse no gráfico.")

    def _on_add_a(self) -> None:
        wf = self._get_active_waveform_base()
        if wf is None:
            self._set_status("Nada para adicionar no Slot A. Desenhe um sinal ou aplique uma fórmula.")
            return
        n_target = self._get_sample_count()
        wf_cycle = self._resample_waveform(wf, n_target) if wf.y.size != n_target else wf
        self._slot_a_cycles = [wf_cycle]
        self._slot_a_base = self._compose_equal_cycles(self._slot_a_cycles, n_target)
        self._refresh_previews()
        self._on_clear()
        self._set_status("Slot A atualizado e gráfico principal limpo.")

    def _on_add_b(self) -> None:
        wf = self._get_active_waveform_base()
        if wf is None:
            self._set_status("Nada para adicionar no Slot B. Desenhe um sinal ou aplique uma fórmula.")
            return
        n_target = self._get_sample_count()
        wf_cycle = self._resample_waveform(wf, n_target) if wf.y.size != n_target else wf
        self._slot_b_cycles = [wf_cycle]
        self._slot_b_base = self._compose_equal_cycles(self._slot_b_cycles, n_target)
        self._refresh_previews()
        self._on_clear()
        self._set_status("Slot B atualizado e gráfico principal limpo.")

    def _on_concat_a(self) -> None:
        wf = self._get_active_waveform_base()
        if wf is None:
            self._set_status("Nada para concatenar no Slot A. Desenhe um sinal ou aplique uma fórmula.")
            return
        n_target = self._get_sample_count()
        wf_cycle = self._resample_waveform(wf, n_target) if wf.y.size != n_target else wf
        if not self._slot_a_cycles:
            self._slot_a_cycles = [wf_cycle]
        else:
            self._slot_a_cycles.append(wf_cycle)
        self._slot_a_base = self._compose_equal_cycles(self._slot_a_cycles, n_target)
        self._refresh_previews()
        self._set_status("Sinal concatenado ao Slot A.")

    def _on_concat_b(self) -> None:
        wf = self._get_active_waveform_base()
        if wf is None:
            self._set_status("Nada para concatenar no Slot B. Desenhe um sinal ou aplique uma fórmula.")
            return
        n_target = self._get_sample_count()
        wf_cycle = self._resample_waveform(wf, n_target) if wf.y.size != n_target else wf
        if not self._slot_b_cycles:
            self._slot_b_cycles = [wf_cycle]
        else:
            self._slot_b_cycles.append(wf_cycle)
        self._slot_b_base = self._compose_equal_cycles(self._slot_b_cycles, n_target)
        self._refresh_previews()
        self._set_status("Sinal concatenado ao Slot B.")

    def _binary_slot_op(self, op_name: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if self._slot_a_base is None or self._slot_b_base is None:
            self._set_status(f"Para {op_name}, preencha Slot A e Slot B usando os botões 'Adicionar' ou 'Concatenar'.")
            return None
        if self._slot_a_base.y.size != self._slot_b_base.y.size:
            self._set_status("Os slots possuem tamanhos diferentes. Ajuste 'Amostras' para o mesmo valor e tente novamente.")
            return None
        return (self._slot_a_base.y, self._slot_b_base.y)

    def _on_sum(self) -> None:
        pair = self._binary_slot_op("somar")
        if pair is None:
            return
        a, b = pair
        y = a + b
        wf = Waveform(x=self._slot_a_base.x, y=y)
        self._current_waveform_base = wf
        self._refresh_main_plot()
        self._set_status("Soma aplicada no gráfico principal. Você pode salvar para .txt.")

    def _on_sub(self) -> None:
        pair = self._binary_slot_op("subtrair")
        if pair is None:
            return
        a, b = pair
        y = a - b
        y = np.clip(y, -1.0, 1.0)
        wf = Waveform(x=self._slot_a_base.x, y=y)
        self._current_waveform_base = wf
        self._refresh_main_plot()
        self._set_status("Subtração aplicada no gráfico principal. Você pode salvar para .txt.")

    def _on_mul(self) -> None:
        pair = self._binary_slot_op("multiplicar")
        if pair is None:
            return
        a, b = pair
        y = a * b
        wf = Waveform(x=self._slot_a_base.x, y=y)
        self._current_waveform_base = wf
        self._refresh_main_plot()
        self._set_status("Multiplicação aplicada no gráfico principal. Você pode salvar para .txt.")

    def _on_div(self) -> None:
        pair = self._binary_slot_op("dividir")
        if pair is None:
            return
        a, b = pair
        eps = 1e-12
        y = a / (b + eps)
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        y = np.clip(y, -1.0, 1.0)
        wf = Waveform(x=self._slot_a_base.x, y=y)
        self._current_waveform_base = wf
        self._refresh_main_plot()
        self._set_status("Divisão aplicada no gráfico principal. Você pode salvar para .txt.")

    def _on_formula_changed(self, text: str) -> None:
        if not self._suppress_custom_switch:
            data = self.preset_combo.currentData()
            if isinstance(data, dict) and data.get("kind") != "custom":
                custom_index = self.preset_combo.findText("Custom")
                if custom_index >= 0:
                    self.preset_combo.setCurrentIndex(custom_index)
        web_engine_view_class = _try_get_web_engine_view_class()
        if web_engine_view_class is None or not isinstance(self.web_view, QtWidgets.QWidget):
            return
        if isinstance(self.web_view, web_engine_view_class):
            latex = self._to_latex(text)
            html = self._mathjax_html(latex)
            self.web_view.setHtml(html)

    def _to_latex(self, expr_text: str) -> str:
        expr_text = (expr_text or "").strip()
        if not expr_text:
            return r"\\text{Digite\ uma\ fórmula,\ por\ exemplo:}\ \sin(2\\pi 10 x)"
        try:
            x = sp.Symbol("x")
            expr = self._safe_sympify(expr_text, x)
            return sp.latex(expr)
        except Exception:
            escaped = (
                expr_text.replace("\\", "\\\\")
                .replace("{", "\\{")
                .replace("}", "\\}")
            )
            return rf"\\text{{Expressão inválida:}}\ {escaped}"

    def _mathjax_html(self, latex: str) -> str:
        # Usa CDN do MathJax; se você precisar offline, dá pra embutir MathJax local depois.
        return f"""<!doctype html>
<html>
<head>
<meta charset=\"utf-8\" />
<script>
  window.MathJax = {{
    tex: {{inlineMath: [['$','$'], ['\\\\(','\\\\)']]}}
  }};
</script>
<script src=\"https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js\"></script>
<style>
  body {{ font-family: sans-serif; margin: 8px; }}
</style>
</head>
<body>
  <div>\\({latex}\\)</div>
</body>
</html>"""

    def _safe_sympify(self, expr_text: str, x: sp.Symbol) -> sp.Expr:
        # Permite um subconjunto de funções comuns, mas evita acesso a builtins perigosos.
        allowed = {
            "x": x,
            "pi": sp.pi,
            "e": sp.E,
            "sin": sp.sin,
            "cos": sp.cos,
            "tan": sp.tan,
            "asin": sp.asin,
            "acos": sp.acos,
            "atan": sp.atan,
            "sinh": sp.sinh,
            "cosh": sp.cosh,
            "tanh": sp.tanh,
            "exp": sp.exp,
            "log": sp.log,
            "sqrt": sp.sqrt,
            "abs": sp.Abs,
            "sign": sp.sign,
            "floor": sp.floor,
            "ceiling": sp.ceiling,
            "Min": sp.Min,
            "Max": sp.Max,
        }
        return sp.sympify(expr_text, locals=allowed)

    def _on_apply_formula(self) -> None:
        data = self.preset_combo.currentData()
        if isinstance(data, dict) and data.get("kind") == "gen":
            try:
                wf = self._generate_preset(str(data.get("gen")))
            except Exception as e:
                self._set_status(f"Erro ao gerar preset: {e}")
                return
        else:
            if isinstance(data, dict) and data.get("kind") == "expr" and data.get("expr_id") in {"sin", "cos"}:
                expr_id = str(data.get("expr_id"))
                fraction = str(self.sine_fraction_combo.currentData() or "1")
                denom = self._get_sample_denom()
                expr_text = f"{expr_id}(2*pi*(x/{denom})*({fraction}))"
                self._set_formula_text_programmatically(expr_text)
            elif isinstance(data, dict) and data.get("kind") == "expr":
                denom = self._get_sample_denom()
                expr_text = str(data.get("expr", "")).strip()
                if expr_text:
                    expr_text = expr_text.replace("8191", str(denom))
                    self._set_formula_text_programmatically(expr_text)
            else:
                expr_text = self.formula_edit.text().strip()
            if not expr_text:
                self._set_status("Digite uma fórmula antes de aplicar.")
                return
            try:
                wf = self._waveform_from_formula(expr_text)
            except Exception as e:
                self._set_status(f"Erro ao interpretar fórmula: {e}")
                return

        if isinstance(data, dict) and data.get("kind") in {"expr", "gen"} and self.preset_invert_check.isChecked():
            wf = Waveform(x=wf.x, y=-wf.y)

        self._current_waveform_base = wf
        self._refresh_main_plot()
        self._set_status("Fórmula aplicada e plotada. Você pode salvar para .txt.")

    def _waveform_from_formula(self, expr_text: str) -> Waveform:
        x = sp.Symbol("x")
        expr = self._safe_sympify(expr_text, x)

        if x not in expr.free_symbols:
            n = self._get_sample_count()
            raise ValueError(f"A fórmula deve usar a variável x (x = 0..{n-1}) pelo menos uma vez.")

        n = self._get_sample_count()
        grid = np.linspace(0.0, 1.0, n, endpoint=False)
        sample_index = np.arange(n, dtype=float)
        f = sp.lambdify(x, expr, modules=["numpy", {"Abs": np.abs}])
        y = np.array(f(sample_index), dtype=float)

        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        y = np.clip(y, -1.0, 1.0)
        return Waveform(x=grid, y=y)

    def _waveform_from_drawing(self) -> Optional[Waveform]:
        pts = self.canvas.get_drawn_points()
        if len(pts) < 2:
            return None

        xs = np.array([p[0] for p in pts], dtype=float)
        ys = np.array([p[1] for p in pts], dtype=float)

        if xs[0] > 0.0:
            xs = np.insert(xs, 0, 0.0)
            ys = np.insert(ys, 0, ys[0])
        if xs[-1] < 1.0:
            xs = np.append(xs, 1.0)
            ys = np.append(ys, ys[-1])

        n = self._get_sample_count()
        grid = np.linspace(0.0, 1.0, n, endpoint=False)

        # Remove duplicatas em x (interp exige x estritamente crescente)
        unique_xs, unique_indices = np.unique(xs, return_index=True)
        unique_ys = ys[unique_indices]

        if unique_xs.size < 2:
            return None

        y = np.interp(grid, unique_xs, unique_ys)
        y = np.clip(y, -1.0, 1.0)
        return Waveform(x=grid, y=y)

    def _get_active_waveform_base(self) -> Optional[Waveform]:
        if self._current_waveform_base is not None:
            return self._current_waveform_base
        return self._waveform_from_drawing()

    def _on_save(self) -> None:
        wf_base = self._get_active_waveform_base()
        wf = self._apply_transform(wf_base, clip=True) if wf_base is not None else None
        if wf is None:
            self._set_status("Nada para salvar. Desenhe um sinal ou aplique uma fórmula.")
            return

        column_name, ok = QtWidgets.QInputDialog.getText(
            self,
            "Nome da coluna",
            "Digite o nome da coluna de dados:",
            text="data",
        )
        if not ok:
            return
        column_name = (column_name or "").strip() or "data"

        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Salvar amostras",
            "waveform.txt",
            "Text Files (*.txt);;All Files (*)",
        )
        if not path:
            return

        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(f"{column_name}\n")
                for v in wf.y.tolist():
                    f.write(f"{float(v):.4f}\n")
        except Exception as e:
            self._set_status(f"Falha ao salvar: {e}")
            return

        self._set_status(f"Arquivo salvo: {path} ({self._get_sample_count()} amostras)")

    def _parse_samples_file(self, path: str) -> np.ndarray:
        values: List[float] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                s = (line or "").strip()
                if not s:
                    continue
                if not values and s.lower() == "data":
                    continue
                # aceita formato: "0.123", "0.123,", "0.123;..." ou "0.123  ..."
                token = s.split(",", 1)[0].split(";", 1)[0].split(None, 1)[0]
                try:
                    values.append(float(token))
                except Exception:
                    raise ValueError(f"Linha inválida: {s}")

        if not values:
            raise ValueError("Arquivo sem amostras")

        y = np.array(values, dtype=float)
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        y = np.clip(y, -1.0, 1.0)
        return y

    def _on_load(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Carregar amostras",
            "",
            "Text Files (*.txt);;CSV Files (*.csv);;All Files (*)",
        )
        if not path:
            return

        try:
            y = self._parse_samples_file(path)
        except Exception as e:
            self._set_status(f"Falha ao carregar: {e}")
            return

        n = int(y.size)
        if n < 2:
            self._set_status("Arquivo inválido: poucas amostras")
            return

        if self.sample_count_spin.value() != n:
            # Ajusta automaticamente o campo e deixa o resto do sistema sincronizar.
            self.sample_count_spin.setValue(n)

        grid = np.linspace(0.0, 1.0, n, endpoint=False)
        self._current_waveform_base = Waveform(x=grid, y=y)
        self._refresh_main_plot()
        self._set_status(f"Arquivo carregado: {path} ({n} amostras)")


def main() -> int:
    if _is_truthy_env("AFG_FORCE_SOFTWARE_RENDERING"):
        os.environ.setdefault("QT_OPENGL", "software")
        os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
        os.environ.setdefault("QT_QUICK_BACKEND", "software")
        os.environ.setdefault(
            "QTWEBENGINE_CHROMIUM_FLAGS",
            "--disable-gpu --disable-gpu-compositing --disable-features=Vulkan",
        )
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.resize(1100, 650)
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
