import sys
import subprocess
import os
import io
import base64
import ipywidgets as widgets
from IPython.display import display, clear_output
import numpy as np
import json
import qiskit.qasm2
from qiskit.quantum_info import Statevector
from qiskit import QuantumCircuit
from qiskit import qpy

# Import your viewer classes
from qc_interactive_education_package import InteractiveViewer, ChallengeViewer
from .quantum_library import QuantumCurriculum


def _sanitize_state(state):
    """
    Intelligently intercepts and converts various mathematical data structures
    (Qiskit Statevectors, Numpy Arrays, complex lists) into a JSON-serializable list of strings.
    Safely strips C-level memoryview bindings (like 'Zd' complex arrays) prior to conversion.
    """
    if state is None:
        return None

    # If it is a Qiskit Statevector, extract the underlying Numpy array natively
    if hasattr(state, 'data'):
        state = state.data

    import numpy as np

    # 1. Force the input into a standard NumPy array to resolve any loose memory bindings
    # 2. Flatten it to ensure a 1D vector
    # 3. Use .tolist() to safely unpack C-level 'Zd' types into native Python complex objects
    safe_list = np.asarray(state).flatten().tolist()

    # Convert every element to a string to guarantee JSON serialization across the OS boundary
    return [str(complex(c)) for c in safe_list]


def launch_app(mode="sandbox", num_qubits=None, initial_state=None, target_state=None, show_circuit=True,
               preloaded_circuit=None):
    """
    Launches the master Single Page Application (SPA) in a new browser tab via Voilà.
    Validates execution modes and serializes arguments to cross the subprocess boundary.
    """
    # Strict validation gates for specialized modes
    if mode == "algorithm" and preloaded_circuit is None:
        raise ValueError("Architectural Error: 'algorithm' mode requires a preloaded_circuit to be provided.")
    if mode == "challenge" and (initial_state is None or target_state is None):
        raise ValueError(
            "Architectural Error: 'challenge' mode requires both initial_state and target_state to be explicitly defined.")

    print(f"Initializing Quantum Education Suite SPA ({mode.capitalize()} Mode)...")

    package_dir = os.path.dirname(os.path.abspath(__file__))
    notebook_path = os.path.join(package_dir, "index.ipynb")

    if not os.path.exists(notebook_path):
        print(f"\n❌ ERROR: Could not find 'index.ipynb' at {notebook_path}")
        return

    # Pass the initial arguments to the Voilà environment
    custom_env = os.environ.copy()

    custom_env["VIEWER_MODE"] = mode

    if num_qubits is not None:
        custom_env["VIEWER_QUBITS"] = str(num_qubits)

    # Intercept and sanitize the states before passing them to JSON
    safe_initial = _sanitize_state(initial_state)
    if safe_initial is not None:
        custom_env["VIEWER_INITIAL"] = json.dumps(safe_initial)

    safe_target = _sanitize_state(target_state)
    if safe_target is not None:
        custom_env["VIEWER_TARGET"] = json.dumps(safe_target)

    custom_env["VIEWER_SHOW_CIRCUIT"] = "1" if show_circuit else "0"

    # Serialize the Qiskit circuit to a QASM string if provided
    custom_env["VIEWER_SHOW_CIRCUIT"] = "1" if show_circuit else "0"

    # ==========================================
    # QPY BINARY SERIALIZATION
    # ==========================================
    # Replaces QASM to perfectly preserve custom gates, labels, and metadata dictionaries natively.
    if preloaded_circuit is not None:
        try:
            buf = io.BytesIO()
            qpy.dump(preloaded_circuit, buf)
            # Encode the binary buffer to a Base64 string for safe OS transit
            custom_env["VIEWER_PRELOADED_QPY"] = base64.b64encode(buf.getvalue()).decode('utf-8')
        except Exception as e:
            print(f"Critical Serialization Failure: Could not pass circuit via QPY. {e}")

    print("Starting local server... A browser window will open automatically once ready.")

    command = [
        sys.executable, "-m", "voila",
        notebook_path,
        "--theme=light",
    ]

    try:
        subprocess.run(command, env=custom_env)
    except KeyboardInterrupt:
        print("\nShutting down Quantum Education Suite...")


# ==========================================
# SPA ENTRY POINT APPLICATION
# ==========================================

class QuantumViewer:
    """
    The Single Page Application (SPA) entry point.
    Renders a persistent top menu and dynamically swaps the active viewer below.
    Auto-initializes the workspace if boot parameters are detected.
    """

    def __init__(self, mode="sandbox", num_qubits=None, initial_state=None, target_state=None, show_circuit=True,
                 preloaded_circuit=None):
        # Dedicated output container for the interactive canvases
        self.viewer_output = widgets.Output(layout={'width': '100%', 'margin': '20px 0px 0px 0px'})

        # Load native Qiskit objects directly from memory
        self.algos = QuantumCurriculum.get_algorithms()
        self.challenges = QuantumCurriculum.get_challenges()

        self.title = widgets.HTML(
            "<h1 style='text-align: center; color: #2c3e50; margin-top: 0px;'>Quantum Viewer</h1>")
        self.subtitle = widgets.HTML(
            "<h4 style='text-align: center; color: #7f8c8d; margin-bottom: 20px;'>Select your learning environment</h4>")
        self.header = widgets.VBox([self.title, self.subtitle], layout={'width': '100%'})

        self.tab = widgets.Tab(layout={'width': '500px', 'min_height': '250px'})

        self.tab_sandbox = self._build_sandbox_tab()
        self.tab_algo = self._build_algorithm_tab()
        self.tab_challenge = self._build_challenge_tab()

        self.tab.children = [self.tab_sandbox, self.tab_algo, self.tab_challenge]
        self.tab.titles = ('Sandbox', 'Algorithms', 'Challenges')

        # The persistent top menu
        self.menu_container = widgets.VBox(
            [self.header, self.tab],
            layout=widgets.Layout(align_items='center', justify_content='center', width='100%',
                                  margin='10px 0px 0px 0px')
        )

        # A subtle visual separator to distinguish the menu from the active workspace
        self.divider = widgets.HTML("<hr style='border: 1px solid #e0e0e0; width: 90%; margin: 15px auto;'>")

        # The master application layout
        self.app_container = widgets.VBox([self.menu_container, self.divider, self.viewer_output],
                                          layout={'width': '100%'})

        # Clean the states for pure inline instantiation as well
        initial_state = _sanitize_state(initial_state)
        target_state = _sanitize_state(target_state)

        # ==========================================
        # AUTO-SPAWN WORKSPACE ON BOOT
        # ==========================================
        if num_qubits is not None or preloaded_circuit is not None or initial_state is not None:
            # Safely deduce qubit dimension if omitted but circuit provided
            _nq = num_qubits
            if _nq is None and preloaded_circuit is not None:
                _nq = preloaded_circuit.num_qubits
            elif _nq is None:
                _nq = 3  # Safe default fallback

            with self.viewer_output:
                if mode == "algorithm":
                    viewer = ChallengeViewer(
                        num_qubits=_nq,
                        preloaded_circuit=preloaded_circuit,
                        show_circuit=show_circuit,
                        is_assessment=False
                    )
                elif mode == "challenge":
                    viewer = ChallengeViewer(
                        num_qubits=_nq,
                        initial_state=initial_state,
                        target_state=target_state,
                        preloaded_circuit=preloaded_circuit,
                        show_circuit=show_circuit,
                        is_assessment=True
                    )
                else:  # Fallback strictly to standard sandbox mode
                    viewer = InteractiveViewer(
                        num_qubits=_nq,
                        initial_state=initial_state,
                        preloaded_circuit=preloaded_circuit,
                        show_circuit=show_circuit
                    )

                # Fast-forward the timeline to render the final state instantly if a circuit was provided
                while viewer._redo_circuit_history:
                    viewer._circuit_history.append(viewer.circuit.copy())
                    viewer._action_history.append(viewer._redo_action_history.pop())
                    viewer.circuit = viewer._redo_circuit_history.pop()

                viewer.display()

    def _build_sandbox_tab(self):
        self.sb_qubits = widgets.Dropdown(options=[1, 2, 3, 4, 5, 6, 7, 8, 9], value=3, description='Qubits:')
        self.sb_circuit = widgets.Checkbox(value=True, description='Show Circuit UI', indent=False)
        self.sb_initial = widgets.Dropdown(description='Initial State:', layout={'width': '380px'})

        self._update_state_dropdown({'new': self.sb_qubits.value})
        self.sb_qubits.observe(self._update_state_dropdown, names='value')

        btn = widgets.Button(description="Launch Sandbox", layout={'width': '100%', 'margin': '15px 0px 0px 0px'})
        btn.style.button_color = '#3498db'
        btn.style.text_color = 'white'
        btn.style.font_weight = 'bold'
        btn.on_click(self._launch_sandbox)

        return widgets.VBox([self.sb_qubits, self.sb_initial, self.sb_circuit, btn],
                            layout={'padding': '20px', 'align_items': 'center'})

    def _update_state_dropdown(self, change):
        n = change['new']
        options_dict = {
            "|0...0⟩ (Ground State)": "ground",
            "|+...+⟩ (Equal Superposition)": "superposition"
        }
        if n == 2:
            options_dict["|Φ⁺⟩ = (|00⟩ + |11⟩)/√2 (Bell State)"] = "ghz"
            options_dict["|Ψ⁺⟩ = (|01⟩ + |10⟩)/√2 (Bell / W-State)"] = "w_state"
        elif n > 2:
            options_dict[f"|GHZ⟩ = (|0...0⟩ + |1...1⟩)/√2 ({n}Q)"] = "ghz"
            options_dict[f"|W⟩ = (|100...⟩ + |010...⟩ + ...)/√{n} ({n}Q)"] = "w_state"

        self.sb_states_map = options_dict
        old_val = self.sb_initial.value
        self.sb_initial.options = list(options_dict.keys())

        if old_val in self.sb_initial.options:
            self.sb_initial.value = old_val
        else:
            self.sb_initial.value = list(options_dict.keys())[0]

    def _build_algorithm_tab(self):
        options = list(self.algos.keys()) if self.algos else ["No algorithms loaded"]
        self.algo_dropdown = widgets.Dropdown(options=options, description='Algorithm:', layout={'width': '350px'})

        self.algo_circuit = widgets.Checkbox(value=True, description='Show Circuit UI', indent=False,
                                             layout={'width': 'auto'})
        self.algo_final_state = widgets.Checkbox(value=True, description='Show Final State', indent=False,
                                                 layout={'width': 'auto'})
        self.algo_annotations = widgets.Checkbox(value=True, description='Show Explanations', indent=False,
                                                 layout={'width': 'auto'})

        checkbox_row = widgets.HBox(
            [self.algo_circuit, self.algo_final_state, self.algo_annotations],
            layout={'display': 'flex', 'flex_flow': 'row wrap', 'justify_content': 'center', 'grid_gap': '15px',
                    'margin': '10px 0px'}
        )

        btn = widgets.Button(description="Study Algorithm", layout={'width': '100%', 'margin': '10px 0px 0px 0px'})
        btn.style.button_color = '#9b59b6'
        btn.style.text_color = 'white'
        btn.style.font_weight = 'bold'
        btn.disabled = not bool(self.algos)
        btn.on_click(self._launch_algorithm)

        return widgets.VBox([self.algo_dropdown, checkbox_row, btn],
                            layout={'padding': '20px', 'align_items': 'center'})

    def _build_challenge_tab(self):
        options = list(self.challenges.keys()) if self.challenges else ["No challenges loaded"]
        self.chal_dropdown = widgets.Dropdown(options=options, description='Challenge:', layout={'width': '400px'})

        self.chal_circuit = widgets.Checkbox(value=True, description='Show Circuit UI', indent=False,
                                             layout={'width': 'auto'})
        self.chal_annotations = widgets.Checkbox(value=True, description='Show Hints', indent=False,
                                                 layout={'width': 'auto'})

        checkbox_row = widgets.HBox(
            [self.chal_circuit, self.chal_annotations],
            layout={'display': 'flex', 'flex_flow': 'row wrap', 'justify_content': 'center', 'grid_gap': '15px',
                    'margin': '10px 0px'}
        )

        btn = widgets.Button(description="Start Challenge", layout={'width': '100%', 'margin': '10px 0px 0px 0px'})
        btn.style.button_color = '#e67e22'
        btn.style.text_color = 'white'
        btn.style.font_weight = 'bold'
        btn.disabled = not bool(self.challenges)
        btn.on_click(self._launch_challenge)

        return widgets.VBox([self.chal_dropdown, checkbox_row, btn],
                            layout={'padding': '20px', 'align_items': 'center'})

    def _launch_sandbox(self, b):
        num_qubits = self.sb_qubits.value
        state_key = self.sb_initial.value
        routing_flag = self.sb_states_map[state_key]

        dim = 2 ** num_qubits
        initial_state = [1.0] + [0.0] * (dim - 1)

        preloaded_qc = QuantumCircuit(num_qubits)
        has_circuit = False

        if routing_flag == "superposition":
            preloaded_qc.h(range(num_qubits))
            has_circuit = True
        elif routing_flag == "ghz":
            preloaded_qc.h(0)
            for i in range(num_qubits - 1):
                preloaded_qc.cx(i, i + 1)
            has_circuit = True
        elif routing_flag == "w_state":
            preloaded_qc.x(0)
            for i in range(num_qubits - 1):
                theta = 2 * np.arccos(1.0 / np.sqrt(num_qubits - i))
                preloaded_qc.cry(theta, i, i + 1)
                preloaded_qc.cx(i + 1, i)
            has_circuit = True

        with self.viewer_output:
            clear_output(wait=True)

            if has_circuit:
                viewer = InteractiveViewer(num_qubits=num_qubits, initial_state=initial_state,
                                           preloaded_circuit=preloaded_qc, show_circuit=self.sb_circuit.value)
                while viewer._redo_circuit_history:
                    viewer._circuit_history.append(viewer.circuit.copy())
                    viewer._action_history.append(viewer._redo_action_history.pop())
                    viewer.circuit = viewer._redo_circuit_history.pop()
                viewer.display()
            else:
                viewer = InteractiveViewer(num_qubits=num_qubits, initial_state=initial_state,
                                           show_circuit=self.sb_circuit.value)
                viewer.display()

    def _launch_algorithm(self, b):
        qc_raw = self.algos[self.algo_dropdown.value]

        with self.viewer_output:
            clear_output(wait=True)
            if self.algo_final_state.value:
                viewer = ChallengeViewer(
                    num_qubits=qc_raw.num_qubits,
                    preloaded_circuit=qc_raw,
                    show_circuit=self.algo_circuit.value,
                    show_annotations=self.algo_annotations.value,
                    is_assessment=False
                )
            else:
                viewer = InteractiveViewer(
                    num_qubits=qc_raw.num_qubits,
                    preloaded_circuit=qc_raw,
                    show_circuit=self.algo_circuit.value,
                    show_annotations=self.algo_annotations.value
                )
            viewer.display()

    def _launch_challenge(self, b):
        chal_data = self.challenges[self.chal_dropdown.value]

        with self.viewer_output:
            clear_output(wait=True)
            viewer = ChallengeViewer(
                num_qubits=chal_data.get("num_qubits"),
                initial_state=chal_data.get("initial_state", None),
                target_state=chal_data.get("target_state", None),
                preloaded_circuit=chal_data.get("preloaded_circuit", None),
                show_circuit=self.chal_circuit.value,
                show_annotations=self.chal_annotations.value,
                is_assessment=True
            )
            viewer.display()

    def display(self):
        from IPython.display import display as ipy_display, HTML

        ipy_display(HTML("""
        <script>
        window.MathJax = window.MathJax || {};
        window.MathJax.Hub = window.MathJax.Hub || {Queue: function(){}};
        </script>
        """))

        ipy_display(self.app_container)