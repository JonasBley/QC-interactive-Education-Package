# ----------------------------------------------------------------------------
# Created By: Nikolas Longen, nlongen@rptu.de
# Modified By: Patrick Pfau, ppfau@rptu.de and Jonas Bley, jonas.bley@rptu.de
# Reviewed By: Maximilian Kiefer-Emmanouilidis, maximilian.kiefer@rptu.de
# Created: March 2023
# Project: DCN QuanTUK
# ----------------------------------------------------------------------------

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from io import BytesIO
from base64 import b64encode
import numpy as np


class Visualization:
    """Superclass for all visualizations of quantum computer states.
    This way all visualizations inherit export methods.
    Alle subclasses must implement/overwrite a draw method and should also
    overwrite the __init__ method.
    """

    # FIXED: Added figsize to the base constructor to persist dimensions across redraws
    def __init__(self, simulator, parse_math=True, figsize=(8.0, 6.0), zero_indexed=True, **kwargs):
        self._sim = simulator
        self.fig = None
        self._lastState = None
        mpl.rcParams["text.usetex"] = False
        mpl.rcParams["text.parse_math"] = parse_math
        # common settings
        self._params = {
            "dpi": 300,
            "transparent": True,
            "showValues": False,
            "bitOrder": simulator._bitOrder,
            "figsize": figsize,
            "zero_indexed": zero_indexed
        }

    def exportPNG(self, fname: str, title=""):
        self._export(fname, "png", title)

    def exportPDF(self, fname: str, title=""):
        self._export(fname, "pdf", title)

    def exportSVG(self, fname: str, title=""):
        mpl.rcParams["svg.fonttype"] = "none"  # Export as text and not paths
        self._export(fname, "svg", title)

    def exportBase64(self, formatStr="png"):
        self._redraw()
        buf = BytesIO()

        self.fig.savefig(
            buf,
            format=formatStr,
            bbox_inches='tight',
            pad_inches=0,
            dpi=self._params["dpi"],
            transparent=self._params["transparent"]
        )

        plt.close(self.fig)
        self.fig = None
        self._lastState = None

        return b64encode(buf.getvalue()).decode('utf-8')

    def _q_label(self, math_index):
        """Returns the localized display string given a 0-indexed mathematical qubit index."""
        offset = 0 if self._params.get("zero_indexed", True) else 1
        return f"Qubit {math_index + offset}"

    def _exportBuffer(self, formatStr, title=""):
        buf = BytesIO()
        self._export(buf, formatStr, title)
        return buf.getbuffer()

    def _export(self, target: str, formatStr: str, title: str):
        self._redraw()
        self.fig.suptitle(title)
        self.fig.savefig(
            target,
            format=formatStr,
            bbox_inches="tight",
            pad_inches=0,
            dpi=self._params["dpi"],
            transparent=self._params["transparent"],
        )

    def show(self):
        self._redraw()

        in_jupyter = False
        try:
            from IPython import get_ipython
            ipython_instance = get_ipython()
            if ipython_instance is not None:
                shell = ipython_instance.__class__.__name__
                if shell == 'ZMQInteractiveShell':
                    in_jupyter = True
        except ImportError:
            pass

        if in_jupyter:
            from IPython.display import display, Image
            import io

            buf = io.BytesIO()
            self.fig.savefig(
                buf,
                format='png',
                bbox_inches='tight',
                pad_inches=0,
                dpi=self._params["dpi"],
                transparent=self._params["transparent"]
            )

            plt.close(self.fig)
            self.fig = None
            self._lastState = None

            display(Image(data=buf.getvalue()))
        else:
            plt.show(block=True)

    def _redraw(self):
        if self._lastState != self._sim:
            self._lastState = self._sim.copy()
            self.draw()

    def showMagnPhase(self, show_values: bool):
        self._params.update({"showValues": show_values})

    @classmethod
    def from_qiskit(cls, qiskit_obj, **kwargs):
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import Statevector

        try:
            from .simulator import Simulator
        except ImportError:
            from simulator import Simulator

        if isinstance(qiskit_obj, QuantumCircuit):
            sv = Statevector.from_instruction(qiskit_obj)
        elif isinstance(qiskit_obj, Statevector):
            sv = qiskit_obj
        else:
            raise TypeError("Input must be a Qiskit QuantumCircuit or Statevector.")

        sim = Simulator(sv.num_qubits)
        sim.writeComplex(sv.data)

        # Pass kwargs through to ensure custom figsizes map correctly
        return cls(sim, **kwargs)

    def draw(self):
        pass

    def _createLabel(self, number: int):
        return np.binary_repr(number, width=self._sim._n)

    def hist(self, qubit=None, size=100) -> tuple[np.array, mpl.figure.Figure, mpl.axes.Axes]:
        _, result = self._sim.read(qubit, size)
        histFig = plt.figure(0)
        ax = histFig.subplots()
        ax.hist(result, density=True)
        ax.set_xlabel("Measured state")
        ax.set_ylabel("N")
        ax.set_title(
            f"Measured all qubits {size} times."
            if qubit is None
            else f"Measured qubit {qubit} {size} times."
        )
        return result, histFig, ax


class CircleNotation(Visualization):
    def __init__(self, simulator, **kwargs):
        # Pass kwargs to superclass to catch figsize overrides
        super().__init__(simulator, **kwargs)

        self._params.update(
            {
                "color_edge": "black",
                "color_fill": "#77b6ba",
                "color_phase": "black",
                "width_edge": 0.7,
                "width_phase": 0.7,
                "textsize_register": 10,
                "textsize_magphase": 8,
                "dist_circles": 3,
                "offset_registerLabel": -1.35,
                "offset_registerValues": -2.3,
            }
        )
        self.fig = None

    def draw(self, cols=8):
        if self._sim._n > 6:
            raise NotImplementedError(
                "Circle notation is only implemented for up to 6 qubits."
            )
        self._cols = cols if cols is not None else 2 ** self._sim._n
        circles = 2 ** self._sim._n
        self._c = self._params["dist_circles"]
        x_max = self._c * self._cols
        y_max = self._c * circles / self._cols if circles > self._cols else self._c
        y_max *= 1 if not self._params["showValues"] else 1.5
        xpos = self._c / 2
        ypos = y_max - self._c / 2

        # FIXED: Enforce figsize
        self.fig = plt.figure(layout="compressed", dpi=self._params["dpi"], figsize=self._params["figsize"])
        ax = self.fig.gca()

        val = np.abs(self._sim._register)
        phi = -np.angle(self._sim._register, deg=False).flatten()
        lx, ly = np.sin(phi), np.cos(phi)

        ax.set_xlim([0, x_max])
        ax.set_ylim([0, y_max])
        ax.set_axis_off()
        ax.set_aspect("equal")

        if (self._sim._n < 6):
            factor = 0.8
        else:
            factor = 0.6 if not self._params["showValues"] else 0.4

        ts_reg = self._params["textsize_register"] * factor
        ts_mag = self._params["textsize_magphase"] * factor

        for i in range(2 ** self._sim._n):
            if val[i] > 1e-3:
                fill = mpatches.Circle(
                    (xpos, ypos),
                    radius=val[i],
                    color=self._params["color_fill"],
                    edgecolor=None,
                )
                phase = mlines.Line2D(
                    [xpos, xpos + lx[i]],
                    [ypos, ypos + ly[i]],
                    color=self._params["color_phase"],
                    linewidth=self._params["width_phase"],
                )
                ax.add_artist(fill)
                ax.add_artist(phase)
            ring = mpatches.Circle(
                (xpos, ypos),
                radius=1,
                fill=False,
                edgecolor=self._params["color_edge"],
                linewidth=self._params["width_edge"],
            )
            ax.add_artist(ring)
            label = self._createLabel(i)
            ax.text(
                xpos,
                ypos + self._params["offset_registerLabel"],
                rf"$|{label:s}\rangle$",
                size=ts_reg,
                horizontalalignment="center",
                verticalalignment="center",
            )

            if self._params["showValues"]:
                ax.text(
                    xpos,
                    ypos + self._params["offset_registerValues"],
                    f"{val[i]:+2.3f}\n{np.rad2deg(phi[i]):+2.0f}°",
                    size=ts_mag,
                    horizontalalignment="center",
                    verticalalignment="center",
                )

            xpos += self._c
            if (i + 1) % self._cols == 0:
                xpos = self._c / 2
                ypos -= self._c if not self._params["showValues"] else self._c * 1.5


class DimensionalCircleNotation(Visualization):
    def __init__(self, simulator, parse_math=True, version=2, **kwargs):
        # Pass kwargs to superclass to catch figsize overrides
        super().__init__(simulator, parse_math=parse_math, **kwargs)

        self._params.update({
            'version': version,
            'labels_dirac': True if version == 1 else False,
            'color_edge': 'black',
            'color_bg': 'white',
            'color_fill': '#77b6baff',
            'color_phase': 'black',
            'color_cube': '#8a8a8a',
            'width_edge': .7,
            'width_phase': .7,
            'width_cube': .5,
            'width_textwidth': .1,
            'offset_registerLabel': 1.3,
            'offset_registerValues': .6,
            'textsize_register': 10 * 0.7 ** ((self._sim._n - 3) // 2),
            'textsize_magphase': 8 * 0.7 ** ((self._sim._n - 3) // 2),
            'textsize_axislbl': 10 * 0.7 ** ((self._sim._n - 3) // 2),
            'bloch_outer_radius': 0.8
        })

        self.fig = None
        self._ax = None
        self._val, self._phi = None, None
        self._lx, self._ly = None, None

    def draw(self):
        """Draw Dimensional Circle Notation representation of current
        simulator state.
        """
        self._textStyle = {
            "size": self._params["textsize_register"],
            "horizontalalignment": "center",
            "verticalalignment": "center",
        }
        self._arrowStyle = {
            "width": 0.03 * 0.7 ** ((self._sim._n - 3) // 2),
            "head_width": 0.2 * 0.7 ** ((self._sim._n - 3) // 2),
            "head_length": 0.3 * 0.7 ** ((self._sim._n - 3) // 2),
            "edgecolor": None,
            "facecolor": 'black',
        }
        self._plotStyle = {
            "color": 'black',
            "linewidth": 0.7 ** ((self._sim._n - 3) // 2),
            "linestyle": "solid",
            "zorder": 0.7 ** ((self._sim._n - 3) // 2),
        }

        # FIXED: Enforce figsize dynamically to prevent warping on new frames
        self.fig = plt.figure(layout="compressed", dpi=self._params["dpi"], figsize=self._params["figsize"])
        self._ax = self.fig.gca()
        self._ax.set_axis_off()
        self._ax.set_aspect("equal")

        self._val = np.abs(self._sim._register)
        self._phi = -np.angle(self._sim._register, deg=False).flatten()
        self._lx, self._ly = np.sin(self._phi), np.cos(self._phi)

        self._axis_labels = np.arange(
            1, self._sim._n + 1)[:: self._params["bitOrder"]]

        amount_qubits = self._sim._n

        if not 0 < amount_qubits < 10 or not isinstance(amount_qubits, int):
            raise NotImplementedError(
                "Please enter a valid number between 1 and 9."
            )

        x, y, len_tick = -2, 7, .2

        if amount_qubits >= 1:
            self._coords = np.array([[0, 1], [1, 1]], dtype=float)
            self._coords *= 3.5

            if self._params['version'] == 1:
                x_pos = x + 1
                y_pos = y - 2
                if amount_qubits == 2:
                    x_pos += -0.5
                    y_pos += 0.3
                elif amount_qubits > 2:
                    x_pos += -1.2
                    y_pos += 0.3
                self._ax.text(
                    x_pos + 1.2,
                    y_pos + 0.3,
                    self._q_label(0),
                    **self._textStyle
                )
                self._ax.arrow(x_pos, y_pos, 2.3, 0, **self._arrowStyle)
            else:
                if amount_qubits == 1:
                    self._ax.arrow(x + 0.5, y - 2, 6.3, 0, **self._arrowStyle)
                    y = 5
                elif amount_qubits == 2:
                    self._ax.arrow(x, y - 2, 6.5, 0, **self._arrowStyle)
                    y = 5
                else:
                    self._ax.arrow(x, y, 6.5, 0, **self._arrowStyle)

                tick_y = [y - len_tick, y + len_tick]
                self._ax.plot(
                    [self._coords[0, 0], self._coords[0, 0]],
                    tick_y,
                    **self._plotStyle
                )
                self._ax.text(
                    self._coords[0, 0],
                    y + 2.5 * len_tick,
                    "0",
                    **self._textStyle,
                )
                self._ax.plot(
                    [self._coords[0, 1], self._coords[0, 1]],
                    tick_y,
                    **self._plotStyle,
                )
                self._ax.text(
                    self._coords[0, 1],
                    y + 2.5 * len_tick,
                    "1",
                    **self._textStyle,
                )
                self._ax.text(
                    self._coords[0, 1] / 2,
                    y + 3 * len_tick,
                    self._q_label(0),
                    **self._textStyle,
                )
            if amount_qubits == 1:
                self._ax.set_xlim([-1.6, 5.3])
                self._ax.set_ylim([2.3, 5.5])

        if amount_qubits >= 2:
            self._coords = np.concatenate((self._coords, np.array([[0, 0], [3.5, 0]])))

            if self._params['version'] == 1:
                x_pos = x + 0.35
                y_pos = y - 2.75
                if amount_qubits > 2:
                    x_pos -= 0.7
                self._ax.text(
                    x_pos - 0.15,
                    y_pos,
                    self._q_label(1),
                    **self._textStyle,
                    rotation=90
                )
                self._ax.arrow(x_pos + 0.15, y_pos + 1.05, 0, -2.3, **self._arrowStyle)
            else:
                if amount_qubits == 2:
                    self._ax.arrow(x, y, 0, -6, **self._arrowStyle)
                else:
                    self._ax.arrow(x, y, 0, -8, **self._arrowStyle)
                tick_x = [x - len_tick, x + len_tick]

                self._ax.plot(
                    tick_x,
                    [self._coords[0, 1], self._coords[0, 1]],
                    **self._plotStyle,
                )
                self._ax.text(
                    x - 2.5 * len_tick,
                    self._coords[0, 1],
                    "0",
                    **self._textStyle,
                    rotation=90
                )
                self._ax.plot(
                    tick_x,
                    [self._coords[3, 1], self._coords[3, 1]],
                    **self._plotStyle,
                )
                self._ax.text(
                    x - 2.5 * len_tick,
                    self._coords[3, 1],
                    "1",
                    **self._textStyle,
                    rotation=90
                )
                self._ax.text(
                    x - 3 * len_tick,
                    self._coords[0, 1] / 2,
                    self._q_label(1),
                    **self._textStyle,
                    rotation=90,
                )
            if amount_qubits == 2:
                self._ax.set_xlim([-2.8, 5])
                self._ax.set_ylim([-1.5, 5.8])

        if amount_qubits >= 3:
            self._coords = np.concatenate((self._coords, self._coords))
            self._coords[4:] += 1.75

            if self._params['version'] == 1:
                self._ax.text(
                    x + 0.55,
                    y - 0.55,
                    self._q_label(2),
                    **self._textStyle,
                    rotation=45
                )
                self._ax.arrow(x - 0.2, y - 1.7, 1.65, 1.65, **self._arrowStyle)
            else:
                self._ax.arrow(x, y, 3.3, 3.3, **self._arrowStyle)
                len_tick_z = len_tick / np.sqrt(2)
                off1, off2 = 0.8, 2.2

                self._ax.plot(
                    [x + off1 + len_tick_z, x + off1 - len_tick_z],
                    [y + off1 - len_tick_z, y + off1 + len_tick_z],
                    **self._plotStyle,
                )
                self._ax.text(
                    x + off1 - 2.5 * len_tick_z,
                    y + off1 + 2.5 * len_tick_z,
                    "0",
                    **self._textStyle,
                    rotation=45
                )
                self._ax.plot(
                    [x + off2 + len_tick_z, x + off2 - len_tick_z],
                    [y + off2 - len_tick_z, y + off2 + len_tick_z],
                    **self._plotStyle,
                )
                self._ax.text(
                    x + off2 - 2.5 * len_tick_z,
                    y + off2 + 2.5 * len_tick_z,
                    "1",
                    **self._textStyle,
                    rotation=45
                )
                middle_ticks = (off2 - off1) / 2
                self._ax.text(
                    x + off1 + (middle_ticks) - 5.5 * len_tick_z,
                    y + off1 + (middle_ticks) + 5.5 * len_tick_z,
                    self._q_label(2),
                    **self._textStyle,
                    rotation=45
                )
            if amount_qubits == 3:
                if self._params['version'] == 1:
                    self._ax.set_ylim([-1.5, 7.5])
                else:
                    self._ax.set_ylim([-1.5, 10.8])
                self._ax.set_xlim([-4.8, 8.5])

        if amount_qubits >= 4:
            for i in range(4, amount_qubits + 1):
                quarter_axis_length = (2 ** int(i / 2))
                self._coords = np.concatenate((self._coords, self._coords))
                if (i % 2 == 0):
                    self._coords[len(self._coords) // 2:, 0] += 2 ** (i / 2 + 1)
                    self._ax.arrow(x, y + i, 4 * quarter_axis_length, 0, **self._arrowStyle)
                    self._ax.plot(
                        [x + quarter_axis_length / 6, x + quarter_axis_length * 1.875],
                        [y + i - 0.3 * len_tick, y + i - 0.3 * len_tick],
                        color='black',
                        linewidth=2,
                        linestyle="solid",
                        zorder=1
                    )
                    self._ax.text(
                        x + quarter_axis_length,
                        y + i + 2.5 * len_tick,
                        "0",
                        **self._textStyle,
                    )
                    self._ax.plot(
                        [x + 2.125 * quarter_axis_length, x + quarter_axis_length * 3.875],
                        [y + i - 0.3 * len_tick, y + i - 0.3 * len_tick],
                        color='black',
                        linewidth=2,
                        linestyle="solid",
                        zorder=1
                    )
                    self._ax.text(
                        3 * quarter_axis_length - 2,
                        y + i + 2.5 * len_tick,
                        "1",
                        **self._textStyle,
                    )
                    self._ax.text(
                        2 * quarter_axis_length - 2,
                        y + i + 3 * len_tick,
                        self._q_label(i-1),
                        **self._textStyle,
                    )
                else:
                    self._coords[len(self._coords) // 2:, 1] -= 2 ** ((i + 1) / 2)
                    x_pos = x + 3 - i
                    self._ax.arrow(x_pos, y, 0, -4 * quarter_axis_length, **self._arrowStyle)
                    self._ax.plot(
                        [x_pos + 0.3 * len_tick, x_pos + 0.3 * len_tick],
                        [y - quarter_axis_length / 6, y - quarter_axis_length * 1.875],
                        color='black',
                        linewidth=2,
                        linestyle="solid",
                        zorder=1
                    )
                    self._ax.text(
                        x_pos - 2.5 * len_tick,
                        y - quarter_axis_length,
                        "0",
                        **self._textStyle,
                        rotation=90,
                    )
                    self._ax.plot(
                        [x_pos + 0.3 * len_tick, x_pos + 0.3 * len_tick],
                        [y - 2.125 * quarter_axis_length, y - quarter_axis_length * 3.875],
                        color='black',
                        linewidth=2,
                        linestyle="solid",
                        zorder=1
                    )
                    self._ax.text(
                        x_pos - 2.5 * len_tick,
                        y - 3 * quarter_axis_length,
                        "1",
                        **self._textStyle,
                        rotation=90,
                    )
                    self._ax.text(
                        x_pos - 3 * len_tick,
                        y - 2 * quarter_axis_length,
                        self._q_label(i-1),
                        **self._textStyle,
                        rotation=90,
                    )
            self._ax.set_xlim([x - 1 - 2 * ((amount_qubits - 3) // 2), x + 1 + 2 ** (amount_qubits // 2 + 2)])
            self._ax.set_ylim([y - 1 - 2 ** ((amount_qubits + 1) // 2 + 1), y + 1 + 2 * ((amount_qubits) // 2)])

        self.draw_all_circles(amount_qubits)

        self._axis_labels = np.arange(1, amount_qubits + 1
                                      )[:: self._params["bitOrder"]]

    def draw_all_circles(self, amount_qubits=None):
        if amount_qubits < 3:
            self._drawLine([0, 1])
        if amount_qubits == 2:
            self._drawLine([0, 2, 3, 1])
        for i in range(2 ** amount_qubits):
            if (i % 8 == 0 and amount_qubits > 2):
                self._drawLine([i, i + 4, i + 5])
                self._drawLine([i + 1, i + 5, i + 7, i + 3])
                self._drawLine([i, i + 2, i + 3, i + 1])
                self._drawLine([i, i + 1])
                self._drawDottedLine([i + 2, i + 6, i + 7])
                self._drawDottedLine([i + 4, i + 6])
            self._drawCircle(i)

    def _drawDottedLine(self, index):
        self._ax.plot(
            self._coords[index, 0],
            self._coords[index, 1],
            color=self._params["color_cube"],
            linewidth=self._params["width_cube"],
            linestyle="dotted",
            zorder=1,
        )

    def _drawLine(self, index):
        self._ax.plot(
            self._coords[index, 0],
            self._coords[index, 1],
            color=self._params["color_cube"],
            linewidth=self._params["width_cube"],
            linestyle="solid",
            zorder=1,
        )

    def _drawCircle(self, index):
        xpos, ypos = self._coords[index]
        bg = mpatches.Circle(
            (xpos, ypos),
            radius=1,
            color=self._params["color_bg"],
            edgecolor=None
        )
        self._ax.add_artist(bg)

        if self._val[index] >= 1e-3:
            fill = mpatches.Circle(
                (xpos, ypos),
                radius=self._val[index],
                color=self._params["color_fill"],
                edgecolor=None,
            )
            self._ax.add_artist(fill)

        ring = mpatches.Circle(
            (xpos, ypos),
            radius=1,
            fill=False,
            edgecolor=self._params["color_edge"],
            linewidth=self._params["width_edge"],
        )
        self._ax.add_artist(ring)

        if self._val[index] >= 1e-3:
            phase = mlines.Line2D(
                [xpos, xpos + self._lx[index]],
                [ypos, ypos + self._ly[index]],
                color=self._params["color_phase"],
                linewidth=self._params["width_phase"],
            )
            self._ax.add_artist(phase)

        label = self._createLabel(index)
        if self._sim._n == 3:
            place = -1 if int(label[1]) else 1
        elif self._sim._n == 2:
            place = -1 if int(label[0]) else 1
        else:
            place = -1

        if self._params['labels_dirac']:
            self._ax.text(
                xpos,
                ypos + place * self._params["offset_registerLabel"],
                rf"$|{label:s}\rangle$",
                **self._textStyle,
            )
        if self._params["showValues"]:
            self._ax.text(
                xpos,
                ypos
                + place
                * (
                        self._params["offset_registerLabel"]
                        + self._params["offset_registerValues"]
                ),
                f"{self._val[index]:+2.3f}\n"
                + f"{np.rad2deg(self._phi[index]):+2.0f}°",
                size=self._params["textsize_magphase"],
                horizontalalignment="center",
                verticalalignment="center",
            )