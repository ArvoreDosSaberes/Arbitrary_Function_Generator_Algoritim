import sys
import math
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import sympy as sp

from PySide6 import QtCore, QtWidgets

try:
    if os.environ.get("AFG_DISABLE_WEBENGINE", "").strip() in {"1", "true", "True", "yes", "YES"}:
        raise ImportError("QtWebEngine disabled by AFG_DISABLE_WEBENGINE")
    from PySide6.QtWebEngineWidgets import QWebEngineView  # type: ignore
except Exception:  # pragma: no cover
    QWebEngineView = None  # type: ignore

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
        self.sum_btn = QtWidgets.QPushButton("Somar A + B", self)

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

        self.sample_count_spin = QtWidgets.QSpinBox(self)
        self.sample_count_spin.setRange(16, 262144)
        self.sample_count_spin.setSingleStep(256)
        self.sample_count_spin.setValue(DEFAULT_N_SAMPLES)

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
        if QWebEngineView is not None:
            self.web_view = QWebEngineView(self)
            self.web_view.setMinimumHeight(120)
        else:
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
        controls.addSpacing(8)
        controls.addWidget(self.save_btn)
        controls.addStretch(1)
        controls.addWidget(self.status)

        left_panel = QtWidgets.QVBoxLayout()
        left_panel.addWidget(QtWidgets.QLabel("Slot A", self))
        left_panel.addWidget(self.preview_a)
        left_panel.addWidget(self.add_a_btn)
        left_panel.addSpacing(10)
        left_panel.addWidget(self.sum_btn)
        left_panel.addSpacing(10)
        left_panel.addWidget(QtWidgets.QLabel("Slot B", self))
        left_panel.addWidget(self.preview_b)
        left_panel.addWidget(self.add_b_btn)
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
        self.save_btn.clicked.connect(self._on_save)
        self.formula_edit.textChanged.connect(self._on_formula_changed)
        self.preset_combo.currentIndexChanged.connect(self._on_preset_changed)
        self.preset_invert_check.stateChanged.connect(self._on_preset_invert_changed)
        self.sine_fraction_combo.currentIndexChanged.connect(self._on_sine_fraction_changed)
        self.triangle_opening_angle_spin.valueChanged.connect(self._on_preset_params_changed)
        self.bell_width_spin.valueChanged.connect(self._on_preset_params_changed)
        self.bell_rise_smooth_spin.valueChanged.connect(self._on_preset_params_changed)
        self.bell_fall_smooth_spin.valueChanged.connect(self._on_preset_params_changed)
        self.exp_log_smooth_spin.valueChanged.connect(self._on_exp_log_smooth_changed)
        self.heartbeat_cycles_spin.valueChanged.connect(self._on_heartbeat_params_changed)
        self.heartbeat_type_combo.currentIndexChanged.connect(self._on_heartbeat_params_changed)
        self.sample_count_spin.valueChanged.connect(self._on_sample_count_changed)
        self.amp_spin.valueChanged.connect(self._on_transform_changed)
        self.phase_spin.valueChanged.connect(self._on_transform_changed)
        self.offset_spin.valueChanged.connect(self._on_transform_changed)

        self.add_a_btn.clicked.connect(self._on_add_a)
        self.add_b_btn.clicked.connect(self._on_add_b)
        self.sum_btn.clicked.connect(self._on_sum)

        self._current_waveform_base: Optional[Waveform] = None
        self._slot_a_base: Optional[Waveform] = None
        self._slot_b_base: Optional[Waveform] = None
        self.preview_a.set_waveform(None)
        self.preview_b.set_waveform(None)
        self._on_formula_changed(self.formula_edit.text())
        self._update_heartbeat_controls()
        self._update_preset_invert_control()
        self._update_sine_controls()
        self._update_exp_log_controls()
        self._update_triangle_controls()
        self._update_bell_controls()

        self._update_save_label()

    def _init_presets(self) -> None:
        # kind: 'expr' usa SymPy; kind: 'gen' gera diretamente 8192 amostras
        self.preset_combo.clear()
        self.preset_combo.addItem("Custom", {"kind": "custom"})

        self.preset_combo.addItem("Seno (sin)", {"kind": "expr", "expr_id": "sin", "expr": "sin(2*pi*x/8191)"})
        self.preset_combo.addItem("Cosseno (cos)", {"kind": "expr", "expr_id": "cos", "expr": "cos(2*pi*x/8191)"})
        self.preset_combo.addItem("Exponencial (exp)", {"kind": "expr", "expr_id": "exp", "expr": "(exp(5*(x/8191)) - 1) / (exp(5) - 1)"})
        self.preset_combo.addItem(
            "Logaritmo natural (ln)",
            {"kind": "expr", "expr_id": "log", "expr": "log(1 + 9*(x/8191)) / log(10)"},
        )

        self.preset_combo.addItem("Quadrada", {"kind": "expr", "expr": "sign(sin(2*pi*x/8191))"})
        self.preset_combo.addItem("Dente de serra", {"kind": "expr", "expr": "x/8191"})
        self.preset_combo.addItem("Rampa", {"kind": "expr", "expr": "x/8191"})
        self.preset_combo.addItem("Triangular", {"kind": "gen", "gen": "triangular"})

        self.preset_combo.addItem("Sinc", {"kind": "gen", "gen": "sinc"})
        self.preset_combo.addItem("Lorenz", {"kind": "gen", "gen": "lorenz"})

        self.preset_combo.addItem("Sino comum (Gaussiana)", {"kind": "gen", "gen": "gaussian_bell"})
        self.preset_combo.addItem("Sino invertido", {"kind": "gen", "gen": "gaussian_bell_inverted"})

        self.preset_combo.addItem("Ruído branco", {"kind": "gen", "gen": "white_noise"})
        self.preset_combo.addItem("Ruído rosa", {"kind": "gen", "gen": "pink_noise"})
        self.preset_combo.addItem("Heartbeat", {"kind": "gen", "gen": "heartbeat"})

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

    def _on_sine_fraction_changed(self, _index: int) -> None:
        # Atualiza o campo de fórmula somente se o preset atual for seno/cosseno.
        data = self.preset_combo.currentData()
        if not (isinstance(data, dict) and data.get("kind") == "expr" and data.get("expr_id") in {"sin", "cos"}):
            return
        expr_id = str(data.get("expr_id"))
        fraction = str(self.sine_fraction_combo.currentData() or "1")
        denom = self._get_sample_denom()
        self.formula_edit.setText(f"{expr_id}(2*pi*(x/{denom})*({fraction}))")

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
            self.formula_edit.setText(f"(exp({k}*(x/{denom})) - 1) / (exp({k}) - 1)")
        else:
            self.formula_edit.setText(f"log(1 + {k}*(x/{denom})) / log(1 + {k})")

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
                self.formula_edit.setText(expr)
        self._update_heartbeat_controls()
        self._update_preset_invert_control()
        self._update_sine_controls()
        self._update_exp_log_controls()
        self._update_triangle_controls()
        self._update_bell_controls()

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

    def _update_save_label(self) -> None:
        n = self._get_sample_count()
        self.save_btn.setText(f"Salvar .txt ({n} amostras)")

    def _resample_waveform(self, wf: Waveform, n_new: int) -> Waveform:
        grid_new = np.linspace(0.0, 1.0, n_new, endpoint=False)
        y_new = np.interp(grid_new, wf.x, wf.y)
        return Waveform(x=grid_new, y=y_new)

    def _on_sample_count_changed(self) -> None:
        n = self._get_sample_count()

        if self._current_waveform_base is not None and self._current_waveform_base.y.size != n:
            self._current_waveform_base = self._resample_waveform(self._current_waveform_base, n)

        if self._slot_a_base is not None and self._slot_a_base.y.size != n:
            self._slot_a_base = self._resample_waveform(self._slot_a_base, n)

        if self._slot_b_base is not None and self._slot_b_base.y.size != n:
            self._slot_b_base = self._resample_waveform(self._slot_b_base, n)

        self._update_save_label()
        self._refresh_previews()
        if self._current_waveform_base is not None:
            self._refresh_main_plot()

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

        if gen_name == "white_noise":
            y = np.random.normal(0.0, 0.35, size=n)
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
        y = np.clip(y, -1.0, 1.0)
        return Waveform(x=grid, y=y)

    def _set_status(self, msg: str) -> None:
        self.status.setText(msg)

    def _apply_transform(self, wf: Waveform, *, clip: bool) -> Waveform:
        y = wf.y

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
        self._slot_a_base = wf
        self._refresh_previews()
        self._on_clear()
        self._set_status("Slot A atualizado e gráfico principal limpo.")

    def _on_add_b(self) -> None:
        wf = self._get_active_waveform_base()
        if wf is None:
            self._set_status("Nada para adicionar no Slot B. Desenhe um sinal ou aplique uma fórmula.")
            return
        self._slot_b_base = wf
        self._refresh_previews()
        self._on_clear()
        self._set_status("Slot B atualizado e gráfico principal limpo.")

    def _on_sum(self) -> None:
        if self._slot_a_base is None or self._slot_b_base is None:
            self._set_status("Para somar, preencha Slot A e Slot B usando os botões 'Adicionar'.")
            return
        if self._slot_a_base.y.size != self._slot_b_base.y.size:
            self._set_status("Os slots possuem tamanhos diferentes. Ajuste 'Amostras' para o mesmo valor e tente novamente.")
            return

        y = self._slot_a_base.y + self._slot_b_base.y
        y = np.clip(y, -1.0, 1.0)
        wf = Waveform(x=self._slot_a_base.x, y=y)
        self._current_waveform_base = wf
        self._refresh_main_plot()
        self._set_status("Soma aplicada no gráfico principal. Você pode salvar para .txt.")

    def _on_formula_changed(self, text: str) -> None:
        if QWebEngineView is None or not isinstance(self.web_view, QtWidgets.QWidget):
            return
        if QWebEngineView is not None and isinstance(self.web_view, QWebEngineView):
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
                self.formula_edit.setText(expr_text)
            elif isinstance(data, dict) and data.get("kind") == "expr":
                denom = self._get_sample_denom()
                expr_text = str(data.get("expr", "")).strip()
                if expr_text:
                    expr_text = expr_text.replace("8191", str(denom))
                    self.formula_edit.setText(expr_text)
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
                f.write("data\n")
                for v in wf.y.tolist():
                    f.write(f"{v}\n")
        except Exception as e:
            self._set_status(f"Falha ao salvar: {e}")
            return

        self._set_status(f"Arquivo salvo: {path} ({self._get_sample_count()} amostras)")


def main() -> int:
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.resize(1100, 650)
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
