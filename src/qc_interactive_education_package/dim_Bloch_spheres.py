import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import hsluv

from qc_interactive_education_package import Simulator, Visualization


class BlochSphere:
    def __init__(self,
                 bloch_radius=0.8,
                 rotation_angle=0,
                 vector_theta=0,
                 vector_phi=0,
                 outer_radius=1,
                 rotation_axis='z'):
        """
        Initialize the BlochSphere.
        """
        self.bloch_radius = bloch_radius
        self.rotation_angle = rotation_angle
        self.vector_theta = vector_theta
        self.vector_phi = vector_phi
        self.outer_radius = outer_radius
        self.rotation_axis = rotation_axis

    def plot(self, ax=None, figsize=(6, 6), offset=(0, 0, 0), fontsize=10):
        """
        Plot the Bloch sphere with everything rotated 90° clockwise about z.
        """
        if ax is None:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection='3d')

        dx, dy, dz = offset

        # Create spherical coordinates.
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 10)

        # --- Outer sphere (low zorder) ---
        x_outer = self.outer_radius * np.outer(np.cos(u), np.sin(v))
        y_outer = self.outer_radius * np.outer(np.sin(u), np.sin(v))
        z_outer = self.outer_radius * np.outer(np.ones_like(u), np.cos(v))

        # Rotate 90° CW: (x, y) -> (y, -x)
        x_outer, y_outer = y_outer, -x_outer

        ax.plot_surface(
            x_outer + dx, y_outer + dy, z_outer + dz,
            color='gray', alpha=0.05, edgecolor='none'
        )

        # --- Draw Equators for 3D depth perception ---
        u_eq = np.linspace(0, 2 * np.pi, 100)

        # 1. X-Y Plane Equator (rotated)
        x_eq_xy = self.outer_radius * np.sin(u_eq)
        y_eq_xy = -self.outer_radius * np.cos(u_eq)
        ax.plot(
            x_eq_xy + dx, y_eq_xy + dy, np.zeros_like(u_eq) + dz,
            color='gray', linestyle='-', alpha=0.5, linewidth=0.5
        )

        # 2. X-Z Plane Equator (rotated: x=0, y=-x_original)
        x_eq_xz = np.zeros_like(u_eq)
        y_eq_xz = -self.outer_radius * np.cos(u_eq)
        z_eq_xz = self.outer_radius * np.sin(u_eq)
        ax.plot(
            x_eq_xz + dx, y_eq_xz + dy, z_eq_xz + dz,
            color='gray', linestyle='-', alpha=0.5, linewidth=0.5
        )

        if self.bloch_radius > 1e-3:
            # --- Internal Bloch vector (Fixed to outer_radius) ---
            x_raw = self.outer_radius * np.sin(self.vector_theta) * np.cos(self.vector_phi)
            y_raw = self.outer_radius * np.sin(self.vector_theta) * np.sin(self.vector_phi)
            z_vec = self.outer_radius * np.cos(self.vector_theta)

            # Rotate the vector too: (x, y) -> (y, -x)
            x_vec, y_vec = y_raw, -x_raw

            # --- Fixed coordinate axes (low zorder) ---
            axes = np.array([
                [self.outer_radius, 0, 0],
                [0, self.outer_radius, 0],
                [0, 0, self.outer_radius]
            ])

            # Draw X, Y, Z axes (rotated)
            rot_axes = np.array([[ay, -ax, az] for ax, ay, az in axes])
            axis_labels = ['x', 'y', 'z']

            for vec, label in zip(rot_axes, axis_labels):
                ax.quiver(
                    dx, dy, dz,
                    vec[0], vec[1], vec[2],
                    color='gray', arrow_length_ratio=0.15, alpha=0.6, linewidth=1
                )
                ax.text(
                    dx + vec[0] * 1.15, dy + vec[1] * 1.15, dz + vec[2] * 1.15,
                    label, color='gray', fontsize=fontsize,
                    ha='center', va='center'
                )

            # Draw the Bloch vector
            ax.quiver(
                dx, dy, dz,
                x_vec, y_vec, z_vec,
                color='#e31b4c', arrow_length_ratio=0.15, linewidth=1.5
            )

            # Dynamic vector label
            ax.text(
                dx + 1.15 * x_vec, dy + 1.15 * y_vec, dz + 1.15 * z_vec,
                r'$v$', color='#e31b4c', fontsize=fontsize,
                ha='left', va='bottom', fontweight='bold'
            )

            # --- Inner sphere (high zorder, transparent) ---
            x_inner = self.bloch_radius * np.outer(np.cos(u), np.sin(v))
            y_inner = self.bloch_radius * np.outer(np.sin(u), np.sin(v))
            z_inner = self.bloch_radius * np.outer(np.ones_like(u), np.cos(v))

            # Rotate the inner sphere
            x_inner, y_inner = y_inner, -x_inner

            # --- HUSL COLOR MAPPING LOGIC ---
            phase_radians = (self.rotation_angle + 5 * np.pi / 4) % (2 * np.pi)
            degrees = np.degrees(phase_radians)
            inner_color = hsluv.hsluv_to_hex([degrees, 100, 50])

            ax.plot_surface(
                x_inner + dx, y_inner + dy, z_inner + dz,
                color=inner_color, alpha=0.3, edgecolor='none'
            )

        # Keep aspect ratio equal
        ax.set_box_aspect([1, 1, 1])
        ax.axis('off')

        return ax


def normalize_vector(v):
    v = np.array(v, dtype=complex)
    norm = np.sqrt(np.sum(np.abs(v) ** 2))
    if norm == 0:
        return v, norm
    return (v / norm, norm)


class SphereNotation(Visualization):
    def __init__(self, simulator, select_qubit=1, parse_math=True, version=2):
        super().__init__(simulator)

        print(f"Setting up DCN Visualization in version {version}.")

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
            'bloch_outer_radius': 1
        })

        self._arrowStyle = {
            "width": 0.03 * 0.7 ** ((self._sim._n - 3) // 2),
            "head_width": 0.2 * 0.7 ** ((self._sim._n - 3) // 2),
            "head_length": 0.3 * 0.7 ** ((self._sim._n - 3) // 2),
            "edgecolor": None,
            "facecolor": 'black',
        }
        self._textStyle = {
            "size": self._params["textsize_register"],
            "horizontalalignment": "center",
            "verticalalignment": "center",
        }
        self._plotStyle = {
            "color": 'black',
            "linewidth": 0.7 ** ((self._sim._n - 3) // 2),
            "linestyle": "solid",
            "zorder": 0.7 ** ((self._sim._n - 3) // 2),
        }
        self.fig = None
        self._ax = None
        self._val, self._phi = None, None
        self._bloch_values = None
        self.select_qubit = select_qubit
        self._lx, self._ly = None, None

        self.husl_cmap = ListedColormap([hsluv.hsluv_to_rgb([h, 100, 50]) for h in np.linspace(0, 360, 256)],
                                        name='husl_phase')

    def draw(self):
        self.fig = plt.figure(layout="compressed")
        plt.get_current_fig_manager().set_window_title("Dimensional Bloch spheres")
        self._ax = self.fig.gca()
        self._ax.set_aspect("equal")

        self._val = np.abs(self._sim._register)
        self._phi = -np.angle(self._sim._register, deg=False).flatten()
        self._lx, self._ly = np.sin(self._phi), np.cos(self._phi)
        self._axis_labels = np.arange(1, self._sim._n + 1)[:: self._params["bitOrder"]]

        amount_qubits = self._sim._n
        select_qubit = self.select_qubit
        register_as_vector = self._sim._register.flatten()
        d = 3.5

        self._bloch_values = multi_complex_to_Bloch(amount_qubits, select_qubit, register_as_vector,
                                                    bitorder=self._params["bitOrder"])

        if not 0 < amount_qubits < 10 or not isinstance(amount_qubits, int):
            raise NotImplementedError("Please enter a valid number between 1 and 9.")

        x, y, len_tick = -2, 7, .2

        if amount_qubits >= 1:
            self._coords = np.array([[0, 1], [1, 1]], dtype=float)
            self._bloch_coords = np.array([[0.0, 1]], dtype=float)
            self._coords *= d
            self._bloch_coords *= d

            if self._params['version'] == 1:
                x_pos = x + 1
                y_pos = y - 2
                if amount_qubits == 2:
                    x_pos += -0.5
                    y_pos += 0.3
                elif amount_qubits > 2:
                    x_pos += -1.2
                    y_pos += 0.3
                self._ax.text(x_pos + 1.2, y_pos + 0.3, "Qubit 1", **self._textStyle)
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
                self._ax.plot([self._coords[0, 0], self._coords[0, 0]], tick_y, **self._plotStyle)
                self._ax.text(self._coords[0, 0], y + 2.5 * len_tick, "0", **self._textStyle)
                self._ax.plot([self._coords[0, 1], self._coords[0, 1]], tick_y, **self._plotStyle)
                self._ax.text(self._coords[0, 1], y + 2.5 * len_tick, "1", **self._textStyle)
                self._ax.text(self._coords[0, 1] / 2, y + 3 * len_tick, "Qubit 1", **self._textStyle)

            if amount_qubits == 1:
                self._ax.set_xlim([-1.6, 5.3])
                self._ax.set_ylim([2.3, 5.5])

        if amount_qubits >= 2:
            self._coords = np.concatenate((self._coords, np.array([[0, 0], [d, 0]])))
            if select_qubit == 1:
                self._bloch_coords = np.array([[0.0, 1], [0.0, 0]], dtype=float)
            else:
                self._bloch_coords = np.array([[0, 1.0], [1.0, 1.0]], dtype=float)

            self._bloch_coords *= d
            if self._params['version'] == 1:
                x_pos = x + 0.35
                y_pos = y - 2.75
                if amount_qubits > 2:
                    x_pos -= 0.7
                self._ax.text(x_pos - 0.15, y_pos, "Qubit 2", **self._textStyle, rotation=90)
                self._ax.arrow(x_pos + 0.15, y_pos + 1.05, 0, -2.3, **self._arrowStyle)
            else:
                if amount_qubits == 2:
                    self._ax.arrow(x, y, 0, -6, **self._arrowStyle)
                else:
                    self._ax.arrow(x, y, 0, -8, **self._arrowStyle)

                tick_x = [x - len_tick, x + len_tick]
                self._ax.plot(tick_x, [self._coords[0, 1], self._coords[0, 1]], **self._plotStyle)
                self._ax.text(x - 2.5 * len_tick, self._coords[0, 1], "0", **self._textStyle, rotation=90)
                self._ax.plot(tick_x, [self._coords[3, 1], self._coords[3, 1]], **self._plotStyle)
                self._ax.text(x - 2.5 * len_tick, self._coords[3, 1], "1", **self._textStyle, rotation=90)
                self._ax.text(x - 3 * len_tick, self._coords[0, 1] / 2, "Qubit 2", **self._textStyle, rotation=90)

            if amount_qubits == 2:
                self._ax.set_xlim([-2.8, 5])
                self._ax.set_ylim([-1.5, 5.8])

        if amount_qubits >= 3:
            self._coords = np.concatenate((self._coords, self._coords))
            self._coords[4:] += d / 2

            if select_qubit == 1 or select_qubit == 2:
                self._bloch_coords = np.concatenate((self._bloch_coords, self._bloch_coords))
                self._bloch_coords[2:] += d / 2
            else:
                self._bloch_coords = np.array([[0, d], [d, d], [0, 0], [d, 0]])

            if self._params['version'] == 1:
                self._ax.text(x + 0.55, y - 0.55, "Qubit 3", **self._textStyle, rotation=45)
                self._ax.arrow(x - 0.2, y - 1.7, d / 2 - 0.1, d / 2 - 0.1, **self._arrowStyle)
            else:
                self._ax.arrow(x, y, d - 0.2, d - 0.2, **self._arrowStyle)
                len_tick_z = len_tick / np.sqrt(2)
                off1, off2 = 0.8, 2.2

                self._ax.plot(
                    [x + off1 + len_tick_z, x + off1 - len_tick_z],
                    [y + off1 - len_tick_z, y + off1 + len_tick_z],
                    **self._plotStyle,
                )
                self._ax.text(x + off1 - 2.5 * len_tick_z, y + off1 + 2.5 * len_tick_z, "0", **self._textStyle,
                              rotation=45)

                self._ax.plot(
                    [x + off2 + len_tick_z, x + off2 - len_tick_z],
                    [y + off2 - len_tick_z, y + off2 + len_tick_z],
                    **self._plotStyle,
                )
                self._ax.text(x + off2 - 2.5 * len_tick_z, y + off2 + 2.5 * len_tick_z, "1", **self._textStyle,
                              rotation=45)

                middle_ticks = (off2 - off1) / 2
                self._ax.text(
                    x + off1 + (middle_ticks) - 5.5 * len_tick_z,
                    y + off1 + (middle_ticks) + 5.5 * len_tick_z,
                    "Qubit 3",
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
            if select_qubit >= 4:
                orig_array = np.array([[0, d], [d, d], [0, 0], [d, 0]])
                self._bloch_coords = np.concatenate((orig_array, orig_array))
                self._bloch_coords[len(self._bloch_coords) // 2:] += d / 2

            for i in range(4, amount_qubits + 1):
                quarter_axis_length = (2 ** int(i / 2))
                self._coords = np.concatenate((self._coords, self._coords))
                if select_qubit < 4:
                    self._bloch_coords = np.concatenate((self._bloch_coords, self._bloch_coords))

                if select_qubit >= 4 and i >= 5:
                    self._bloch_coords = np.concatenate((self._bloch_coords, self._bloch_coords))


                if (i % 2 == 0):
                    self._coords[len(self._coords) // 2:, 0] += 2 ** (i / 2 + 1)

                    if select_qubit < 4:
                        self._bloch_coords[len(self._bloch_coords) // 2:, 0] += 2 ** (i / 2 + 1)
                    elif i >= 6:
                        self._bloch_coords[len(self._bloch_coords) // 2:, 0] += 2 ** (i / 2 + 1)



                    self._ax.arrow(x, y + i, 4 * quarter_axis_length, 0, **self._arrowStyle)
                    self._ax.plot(
                        [x + quarter_axis_length / 6, x + quarter_axis_length * 1.875],
                        [y + i - 0.3 * len_tick, y + i - 0.3 * len_tick],
                        color='black', linewidth=2, linestyle="solid", zorder=1
                    )
                    self._ax.text(x + quarter_axis_length, y + i + 2.5 * len_tick, "0", **self._textStyle)

                    self._ax.plot(
                        [x + 2.125 * quarter_axis_length, x + quarter_axis_length * 3.875],
                        [y + i - 0.3 * len_tick, y + i - 0.3 * len_tick],
                        color='black', linewidth=2, linestyle="solid", zorder=1
                    )
                    self._ax.text(3 * quarter_axis_length - 2, y + i + 2.5 * len_tick, "1", **self._textStyle)
                    self._ax.text(2 * quarter_axis_length - 2, y + i + 3 * len_tick, f"Qubit {i}", **self._textStyle)
                else:
                    self._coords[len(self._coords) // 2:, 1] -= 2 ** ((i + 1) / 2)

                    if select_qubit < 4:
                        self._bloch_coords[len(self._bloch_coords) // 2:, 1] -= 2 ** ((i + 1) / 2)
                    elif select_qubit % 2 == 0:
                        self._bloch_coords[len(self._bloch_coords) // 2:, 1] -= 2 ** ((i + 1) / 2)
                    elif select_qubit % 2 == 1:
                        self._bloch_coords[len(self._bloch_coords) // 2:, 0] += 2 ** ((i + 1) / 2)

                    x_pos = x + 3 - i
                    self._ax.arrow(x_pos, y, 0, -4 * quarter_axis_length, **self._arrowStyle)

                    self._ax.plot(
                        [x_pos + 0.3 * len_tick, x_pos + 0.3 * len_tick],
                        [y - quarter_axis_length / 6, y - quarter_axis_length * 1.875],
                        color='black', linewidth=2, linestyle="solid", zorder=1
                    )
                    self._ax.text(x_pos - 2.5 * len_tick, y - quarter_axis_length, "0", **self._textStyle, rotation=90)

                    self._ax.plot(
                        [x_pos + 0.3 * len_tick, x_pos + 0.3 * len_tick],
                        [y - 2.125 * quarter_axis_length, y - quarter_axis_length * 3.875],
                        color='black', linewidth=2, linestyle="solid", zorder=1
                    )
                    self._ax.text(x_pos - 2.5 * len_tick, y - 3 * quarter_axis_length, "1", **self._textStyle,
                                  rotation=90)
                    self._ax.text(x_pos - 3 * len_tick, y - 2 * quarter_axis_length, f"Qubit {i}", **self._textStyle,
                                  rotation=90)



            self._ax.set_xlim([x - 1 - 2 * ((amount_qubits - 3) // 2), x + 1 + 2 ** (amount_qubits // 2 + 2)])
            self._ax.set_ylim([y - 1 - 2 ** ((amount_qubits + 1) // 2 + 1), y + 1 + 2 * ((amount_qubits) // 2)])

        def swap_middle_quarters_reverse(arr):
            n = len(arr)
            k = n // 4

            # Reverse Q2 in-place using numpy slicing
            arr[k: 2 * k] = arr[k: 2 * k][::-1]
            # Reverse Q3 in-place
            arr[2 * k: 3 * k] = arr[2 * k: 3 * k][::-1]
            # Reverse the combined block
            arr[k: 3 * k] = arr[k: 3 * k][::-1]

            return arr

        def iterate_through_swaps(arr,it_number):
            k = len(arr) // it_number
            for i in range(it_number):
                arr[i*k : (i+1)*k] = swap_middle_quarters_reverse(arr[i*k : (i+1)*k])
            return arr

        if select_qubit == 6:
            if amount_qubits == 6:
                self._bloch_coords = swap_middle_quarters_reverse(self._bloch_coords)
                self._bloch_coords[len(self._bloch_coords) // 2:, 0] -= 2 ** ((5 + 1) / 2)
            if amount_qubits == 7:
                self._bloch_coords[len(self._bloch_coords) // 4:len(self._bloch_coords) // 2, 0] -= 2 ** ((5 + 1) / 2)
                self._bloch_coords[3*len(self._bloch_coords) // 4:, 0] -= 2 ** ((5 + 1) / 2)
                self._bloch_coords = iterate_through_swaps(self._bloch_coords, 2)

                # self._bloch_coords = reverse_inner_chunks(self._bloch_coords, 4)



        self._bloch_coords = self._bloch_coords[::self._params["bitOrder"]]

        # Draw all spheres
        self._draw_all_spheres()

        self._ax.set_axis_off()
        self._axis_labels = np.arange(1, amount_qubits + 1)[:: self._params["bitOrder"]]

        # ==========================================
        # DYNAMIC COLORBAR PLACEMENT & LABELS
        # ==========================================
        try:
            cbar_size = 2.5  # Data units
            if amount_qubits == 2:
                arr_x = x + 6.5
            else:
                quarter_axis_length = (2 ** int(amount_qubits / 2))
                arr_x = x + 4 * quarter_axis_length

            # Center vertically to match the relative 0.45 height used by the odd-qubit layout
            ymin, ymax = self._ax.get_ylim()
            cbar_center_y = ymin + 0.45 * (ymax - ymin)

            cbar_ax = self._ax.inset_axes(
                [arr_x + 0.5, cbar_center_y - cbar_size / 2, cbar_size, cbar_size],
                transform=self._ax.transData,
                projection='polar'
            )

            n_segments = 360
            theta = np.linspace(0, 2 * np.pi, n_segments)
            r = np.linspace(0.6, 1, 2)
            Theta, R = np.meshgrid(theta, r)
            ColorVals = (Theta + 5 * np.pi / 4) % (2 * np.pi)

            mesh = cbar_ax.pcolormesh(Theta, R, ColorVals, cmap=self.husl_cmap, shading='auto', vmin=0, vmax=2 * np.pi)

            cbar_ax.set_yticks([])
            cbar_ax.set_xticks([])  # Clear default ticks to pull them closer manually
            cbar_ax.spines['polar'].set_visible(False)

            # Manual tight label placement
            pad_r = 1.25  # Tightly anchored to the outer ring of the color wheel
            lbl_size = self._params['textsize_magphase']

            cbar_ax.text(0, pad_r, r'$0$', ha='left', va='center', fontsize=lbl_size)
            cbar_ax.text(np.pi / 2, pad_r, r'$\pi/2$', ha='center', va='bottom', fontsize=lbl_size)
            cbar_ax.text(np.pi, pad_r, r'$\pi$', ha='right', va='center', fontsize=lbl_size)
            cbar_ax.text(3 * np.pi / 2, pad_r, r'$3\pi/2$', ha='center', va='top', fontsize=lbl_size)

        except Exception as e:
            print(f"Error creating circular colorbar: {e}")

    def _draw_all_spheres(self):
        if self._bloch_coords is None or self._bloch_values is None:
            return
        if len(self._bloch_coords) != len(self._bloch_values):
            return

        outer_radius = self._params['bloch_outer_radius']
        inset_diameter_data = 2 * outer_radius * 2

        for i in range(len(self._bloch_coords)):
            cx, cy = self._bloch_coords[i]
            bloch_params = self._bloch_values[i]

            try:
                radius = bloch_params[0]
                angle = bloch_params[1]
                theta = bloch_params[2]
                phi = bloch_params[3]
            except (IndexError, TypeError):
                radius, angle, theta, phi = 0.0, 0.0, 0.0, 0.0

            bl_x = cx - inset_diameter_data / 2
            bl_y = cy - inset_diameter_data / 2

            inset_ax = self._ax.inset_axes(
                [bl_x, bl_y, inset_diameter_data, inset_diameter_data],
                transform=self._ax.transData,
                zorder=2,
                projection='3d'
            )

            sphere = BlochSphere(
                bloch_radius=radius,
                rotation_angle=angle,
                vector_theta=theta,
                vector_phi=phi,
                outer_radius=outer_radius
            )

            sphere.plot(ax=inset_ax, fontsize=self._params['textsize_register'])

            pane_color = (1.0, 1.0, 1.0, 0.0)
            inset_ax.set_facecolor(pane_color)
            inset_ax.xaxis.set_pane_color(pane_color)
            inset_ax.yaxis.set_pane_color(pane_color)
            inset_ax.zaxis.set_pane_color(pane_color)

            try:
                inset_ax.xaxis._axinfo["grid"]['color'] = pane_color
                inset_ax.yaxis._axinfo["grid"]['color'] = pane_color
                inset_ax.zaxis._axinfo["grid"]['color'] = pane_color
            except (KeyError, AttributeError):
                pass

            inset_ax.set_axis_off()

    def _get_fixed_state_label(self, index, n_qubits, selected_qubit):
        if n_qubits == 1: return ""
        num_fixed = n_qubits - 1
        binary_format = "{:0" + str(num_fixed) + "b}"
        binary_string = binary_format.format(index)

        label_list = list(binary_string)
        label_list.insert(selected_qubit - 1, '-')
        return "|" + "".join(label_list) + ">"


def complex_to_bloch(vector):
    vector, norm = normalize_vector(vector)
    alpha = complex(vector[0])
    beta = complex(vector[1])

    if np.abs(alpha) > 1e-6:
        global_phase = np.angle(alpha)
    else:
        global_phase = np.angle(beta)

    alpha *= np.exp(-1j * global_phase)
    beta *= np.exp(-1j * global_phase)

    theta = 2 * np.arccos(np.clip(np.real(alpha), 0, 1))
    phi = np.angle(beta)

    return float(norm), float(global_phase), float(theta), float(phi)


def select_qubits(n, sel_qubit):
    reordered_list = []
    dif = int(2 ** (sel_qubit - 1))
    N = int(2 ** (n - 1))

    for i in range(2 * N):
        if i not in reordered_list:
            reordered_list.append(i)
            reordered_list.append(i + dif)

    pairs = [[reordered_list[x * 2], reordered_list[x * 2 + 1]] for x in range(N)]
    return pairs


def multi_complex_to_Bloch(n, sel_qubit, vector, bitorder=1):
    if isinstance(vector, Simulator):
        vector = vector._register.flatten()
    vector = normalize_vector(vector)[0][::bitorder]
    multi_bloch = []
    for pair in select_qubits(n, sel_qubit):
        if vector[pair[0]] == 0 and vector[pair[1]] == 0:
            multi_bloch.append([0, 0, 0, 0])
        else:
            norm, global_phase, theta, phi = complex_to_bloch([vector[pair[0]], vector[pair[1]]])
            multi_bloch.append([norm, global_phase, theta, phi])
    return multi_bloch[::bitorder]