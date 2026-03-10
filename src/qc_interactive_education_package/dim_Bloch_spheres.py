# ----------------------------------------------------------------------------
# Created March 2026 by Jonas Bley, jonas.bley@rptu.de
# ----------------------------------------------------------------------------

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
        self.bloch_radius = bloch_radius
        self.rotation_angle = rotation_angle
        self.vector_theta = vector_theta
        self.vector_phi = vector_phi
        self.outer_radius = outer_radius
        self.rotation_axis = rotation_axis

    def plot(self, ax=None, figsize=(6, 6), offset=(0, 0, 0), fontsize=10):
        if ax is None:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection='3d')

        dx, dy, dz = offset

        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 10)

        x_outer = self.outer_radius * np.outer(np.cos(u), np.sin(v))
        y_outer = self.outer_radius * np.outer(np.sin(u), np.sin(v))
        z_outer = self.outer_radius * np.outer(np.ones_like(u), np.cos(v))

        x_outer, y_outer = y_outer, -x_outer

        ax.plot_surface(
            x_outer + dx, y_outer + dy, z_outer + dz,
            color='gray', alpha=0.05, edgecolor='none'
        )

        u_eq = np.linspace(0, 2 * np.pi, 100)

        x_eq_xy = self.outer_radius * np.sin(u_eq)
        y_eq_xy = -self.outer_radius * np.cos(u_eq)
        ax.plot(
            x_eq_xy + dx, y_eq_xy + dy, np.zeros_like(u_eq) + dz,
            color='gray', linestyle='-', alpha=0.5, linewidth=0.5
        )

        x_eq_xz = np.zeros_like(u_eq)
        y_eq_xz = -self.outer_radius * np.cos(u_eq)
        z_eq_xz = self.outer_radius * np.sin(u_eq)
        ax.plot(
            x_eq_xz + dx, y_eq_xz + dy, z_eq_xz + dz,
            color='gray', linestyle='-', alpha=0.5, linewidth=0.5
        )

        if self.bloch_radius > 1e-3:
            x_raw = self.outer_radius * np.sin(self.vector_theta) * np.cos(self.vector_phi)
            y_raw = self.outer_radius * np.sin(self.vector_theta) * np.sin(self.vector_phi)
            z_vec = self.outer_radius * np.cos(self.vector_theta)

            x_vec, y_vec = y_raw, -x_raw

            axes = np.array([
                [self.outer_radius, 0, 0],
                [0, self.outer_radius, 0],
                [0, 0, self.outer_radius]
            ])

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

            ax.quiver(
                dx, dy, dz,
                x_vec, y_vec, z_vec,
                color='#e31b4c', arrow_length_ratio=0.15, linewidth=1.5
            )

            x_inner = self.bloch_radius * np.outer(np.cos(u), np.sin(v))
            y_inner = self.bloch_radius * np.outer(np.sin(u), np.sin(v))
            z_inner = self.bloch_radius * np.outer(np.ones_like(u), np.cos(v))

            x_inner, y_inner = y_inner, -x_inner

            phase_radians = (self.rotation_angle + 5 * np.pi / 4) % (2 * np.pi)
            degrees = np.degrees(phase_radians)
            inner_color = hsluv.hsluv_to_hex([degrees, 100, 50])

            ax.plot_surface(
                x_inner + dx, y_inner + dy, z_inner + dz,
                color=inner_color, alpha=0.3, edgecolor='none'
            )

        ax.set_box_aspect([1, 1, 1])
        ax.axis('off')
        return ax


def normalize_vector(v):
    v = np.array(v, dtype=complex)
    norm = np.sqrt(np.sum(np.abs(v) ** 2))
    if norm == 0:
        return v, norm
    return (v / norm, norm)


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
    dif = int(2 ** (sel_qubit))
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
        self._bloch_coords = None
        self.select_qubit = select_qubit
        self._lx, self._ly = None, None

        self.husl_cmap = ListedColormap([hsluv.hsluv_to_rgb([h, 100, 50]) for h in np.linspace(0, 360, 256)],
                                        name='husl_phase')

    def draw(self):
        self.fig = plt.figure(layout="compressed")
        plt.get_current_fig_manager().set_window_title("Dimensional Bloch spheres")
        self._ax = self.fig.gca()
        self._ax.set_aspect("equal")
        self._bounding_points = []

        self._val = np.abs(self._sim._register)
        self._phi = -np.angle(self._sim._register, deg=False).flatten()
        self._lx, self._ly = np.sin(self._phi), np.cos(self._phi)

        amount_qubits = self._sim._n
        select_qubit = self.select_qubit
        register_as_vector = self._sim._register.flatten()
        d = 3.5
        x_base, y_base, len_tick = -2, 7, 0.2

        if not 0 < amount_qubits < 10 or not isinstance(amount_qubits, int):
            raise NotImplementedError("Please enter a valid number between 1 and 9.")

        # Obtain the parameters for each sphere
        self._bloch_values = multi_complex_to_Bloch(
            amount_qubits, select_qubit, register_as_vector, bitorder=self._params["bitOrder"]
        )

        # 1. Generate Coordinates Vectorially
        self._generate_coordinates(amount_qubits, select_qubit, d)

        # Ensure bitOrder alignment matches the legacy implementation
        self._bloch_coords = self._bloch_coords[::self._params["bitOrder"]]

        # 2. Draw axes and spheres
        if self._params['version'] != 1:
            self._draw_legend_axes(amount_qubits, select_qubit, d, x_base, y_base, len_tick)

        self._draw_all_spheres()
        self._ax.set_axis_off()

        # 3. Dynamic layout framing (Replaces hardcoded if/elif blocks)
        self._apply_dynamic_framing(amount_qubits, x_base)

        # 4. Draw Colorbar (Uses the newly established y-limits to center itself)
        self._draw_colorbar(amount_qubits, x_base)

    def _generate_coordinates(self, amount_qubits, select_qubit, d):
        """Replaces legacy array slicing/swapping with pure binary state projection."""
        vectors = []
        for k in range(1, amount_qubits + 1):
            if k == 1:
                vectors.append([d, 0])
            elif k == 2:
                vectors.append([0, -d])
            elif k == 3:
                vectors.append([d / 2, d / 2])
            else:
                if k % 2 == 0:
                    vectors.append([2 ** (k / 2 + 1), 0])
                else:
                    vectors.append([0, -2 ** ((k + 1) / 2)])

        vectors = np.array(vectors, dtype=float)

        # Project out the axis of the selected qubit
        spectator_vectors = np.delete(vectors, select_qubit, axis=0)

        num_spectators = amount_qubits - 1
        num_spheres = 2 ** num_spectators
        indices = np.arange(num_spheres)

        # Create binary representations (mapping logic bits to basis vectors)
        binary_matrix = ((indices[:, None] & (1 << np.arange(num_spectators))) > 0).astype(float)

        # Compute coordinates simultaneously
        start_coord = np.array([0.0, d])
        self._bloch_coords = start_coord + binary_matrix.dot(spectator_vectors)

        # ACCUMULATOR FIX: Add the physical boundaries of the spheres (radius = 2.0)
        # to guarantee the camera wraps the spheres tightly without over-padding axes.
        radius = 2.0
        for cx, cy in self._bloch_coords:
            self._bounding_points.extend([
                [cx - radius, cy - radius],
                [cx + radius, cy + radius]
            ])

    def _draw_qubit_axis(self, label, start_pos, vec, tick_len):
        """Universally draws an axis and labels using pure vector math, without redundant line segments."""
        start_pos = np.array(start_pos, dtype=float)
        vec = np.array(vec, dtype=float)

        length = np.linalg.norm(vec)
        if length == 0: return

        direction = vec / length
        # 90-degree counter-clockwise rotation defines the "above" normal
        normal = np.array([-direction[1], direction[0]])

        # Detect if the vector is pointing downwards (Qubit 2, 5, 7, etc.)
        is_downward = (direction[1] < -0.5 and abs(direction[0]) < 0.5)

        # 1. Main Arrow
        self._ax.arrow(start_pos[0], start_pos[1], vec[0], vec[1], **self._arrowStyle)

        # 2. State |0> and |1> text labels
        # Placed slightly below the axis (-normal) to balance the qubit label above


        # 4. Qubit Label Positioning (Pushed past the arrow tip)
        base_label_pos = start_pos + 0.5*vec + direction*(tick_len*1.5)

        if is_downward:
            # Shift to the LEFT (-normal) and rotate to 90 degrees (180 deg turn from -90)
            label_pos = base_label_pos - normal * (tick_len * 2)
            mid_0 = start_pos + 0.30 * vec - normal * (tick_len * 1.5)
            mid_1 = start_pos + 0.80 * vec - normal * (tick_len * 1.5)
            angle = 90
        else:
            # Shift ABOVE (+normal)
            label_pos = base_label_pos + normal * (tick_len * 2)
            mid_0 = start_pos + 0.30 * vec + normal * (tick_len * 1.5)
            mid_1 = start_pos + 0.80 * vec + normal * (tick_len * 1.5)
            angle = np.degrees(np.arctan2(vec[1], vec[0]))
            if angle < -90 or angle > 90:
                angle += 180

        self._ax.text(mid_0[0], mid_0[1], "0", **self._textStyle)
        self._ax.text(mid_1[0], mid_1[1], "1", **self._textStyle)
        self._ax.text(label_pos[0], label_pos[1], label, rotation=angle, **self._textStyle)

        # Accumulate extreme points of the axis for framing
        self._bounding_points.extend([
            start_pos.tolist(),
            (start_pos + vec).tolist(),
            label_pos.tolist()
        ])

    def _apply_dynamic_framing(self, amount_qubits, x_base):
        """Calculates and applies camera limits based on the accumulated geometric point cloud."""
        if not hasattr(self, '_bounding_points') or not self._bounding_points:
            return

        points = np.array(self._bounding_points)
        min_x, max_x = np.min(points[:, 0]), np.max(points[:, 0])
        min_y, max_y = np.min(points[:, 1]), np.max(points[:, 1])

        cbar_arr_x = x_base + 6.5 if amount_qubits == 2 else x_base + 4 * (2 ** int(amount_qubits / 2))
        cbar_right_edge = cbar_arr_x + 3.0
        max_x = max(max_x, cbar_right_edge)

        # REDUCED PADDING: 0.3 just prevents text/lines from touching the absolute pixel edge
        pad_x, pad_y = 0.3, 0.3

        self._ax.set_xlim([min_x - pad_x, max_x + pad_x])
        self._ax.set_ylim([min_y - pad_y, max_y + pad_y])

    def _draw_legend_axes(self, amount_qubits, select_qubit, d, x, y, len_tick):
        # We loop by mathematical index (0 to n-1), but geometry is still based on k
        for idx in range(amount_qubits):
            if idx == select_qubit: continue

            k = idx + 1  # Geometric layout logic remains intact

            if k == 1:
                start = [x + 0.5, y - 2] if amount_qubits == 1 else [x, y - 2] if amount_qubits == 2 else [x, y]
                vec = [6.3, 0] if amount_qubits == 1 else [6.5, 0]
            elif k == 2:
                start = [x, y - 2] if amount_qubits == 2 else [x, y]
                vec = [0, -6.0] if amount_qubits == 2 else [0, -8.0]
            elif k == 3:
                start = [x, y];
                vec = [d - 0.2, d - 0.2]
            else:
                quarter_axis_length = 2 ** (k // 2)
                if k % 2 == 0:
                    start = [x, y + k];
                    vec = [4 * quarter_axis_length, 0]
                else:
                    start = [x + 3 - k, y];
                    vec = [0, -4 * quarter_axis_length]

            # Dynamically fetch the correct string ("Qubit 0" or "Qubit 1")
            self._draw_qubit_axis(self._q_label(idx), start, vec, len_tick)

    def _draw_colorbar(self, amount_qubits, x):
        try:
            cbar_size = 1.5
            arr_x = x + 6.5 if amount_qubits == 2 else x + 4 * (2 ** int(amount_qubits / 2))

            ymin, ymax = self._ax.get_ylim()
            cbar_center_y = ymin + 0.45 * (ymax - ymin)

            cbar_ax = self._ax.inset_axes(
                [arr_x + 0.5, cbar_center_y - cbar_size / 2, cbar_size, cbar_size],
                transform=self._ax.transData,
                projection='polar'
            )

            theta = np.linspace(0, 2 * np.pi, 360)
            r = np.linspace(0.6, 1, 2)
            Theta, R = np.meshgrid(theta, r)
            ColorVals = (Theta + 5 * np.pi / 4) % (2 * np.pi)

            cbar_ax.pcolormesh(Theta, R, ColorVals, cmap=self.husl_cmap, shading='auto', vmin=0, vmax=2 * np.pi)

            cbar_ax.set_yticks([])
            cbar_ax.set_xticks([])
            cbar_ax.spines['polar'].set_visible(False)

            pad_r = 1.25
            lbl_size = self._params['textsize_magphase']

            cbar_ax.text(0, pad_r, r'$0$', ha='left', va='center', fontsize=lbl_size)
            cbar_ax.text(np.pi / 2, pad_r, r'$\pi/2$', ha='center', va='bottom', fontsize=lbl_size)
            cbar_ax.text(np.pi, pad_r, r'$\pi$', ha='right', va='center', fontsize=lbl_size)
            cbar_ax.text(3 * np.pi / 2, pad_r, r'$3\pi/2$', ha='center', va='top', fontsize=lbl_size)
        except Exception as e:
            print(f"Error creating circular colorbar: {e}")

    def _draw_all_spheres(self):
        if self._bloch_coords is None or self._bloch_values is None: return
        if len(self._bloch_coords) != len(self._bloch_values): return

        outer_radius = self._params['bloch_outer_radius']
        inset_diameter_data = 2 * outer_radius * 2

        for i in range(len(self._bloch_coords)):
            cx, cy = self._bloch_coords[i]
            bloch_params = self._bloch_values[i]

            try:
                radius, angle, theta, phi = bloch_params[0], bloch_params[1], bloch_params[2], bloch_params[3]
            except (IndexError, TypeError):
                radius, angle, theta, phi = 0.0, 0.0, 0.0, 0.0

            bl_x, bl_y = cx - inset_diameter_data / 2, cy - inset_diameter_data / 2

            inset_ax = self._ax.inset_axes(
                [bl_x, bl_y, inset_diameter_data, inset_diameter_data],
                transform=self._ax.transData,
                zorder=2,
                projection='3d'
            )

            sphere = BlochSphere(radius, angle, theta, phi, outer_radius)
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
        binary_string = ("{:0" + str(num_fixed) + "b}").format(index)
        label_list = list(binary_string)
        # Insert exactly at the 0-indexed location
        label_list.insert(selected_qubit, '-')
        return "|" + "".join(label_list) + ">"