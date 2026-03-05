import sys
import subprocess
import os
import ipywidgets as widgets
from IPython.display import display, clear_output
import numpy as np
import json
import qiskit.qasm2
from qiskit.quantum_info import Statevector
from qiskit import QuantumCircuit

# Import your viewer classes
from qc_interactive_education_package import InteractiveViewer, ChallengeViewer
from .quantum_library import QuantumCurriculum


def launch_tool(num_qubits=3, initial_state=None, show_circuit=True, preloaded_circuit=None):
    """
    Launches the Voilà server for the interactive quantum sandbox.
    """
    print(f"Initializing Quantum Sandbox Environment ({num_qubits} Qubits)...")

    package_dir = os.path.dirname(os.path.abspath(__file__))
    notebook_path = os.path.join(package_dir, "app.ipynb")

    if not os.path.exists(notebook_path):
        print(f"\n❌ ERROR: Could not find 'app.ipynb' at {notebook_path}")
        return

    custom_env = os.environ.copy()
    custom_env["VIEWER_QUBITS"] = str(num_qubits)
    custom_env["VIEWER_INITIAL"] = json.dumps(initial_state)
    custom_env["VIEWER_SHOW_CIRCUIT"] = "1" if show_circuit else "0"

    # NEW: Serialize the Qiskit circuit to a QASM string
    if preloaded_circuit is not None:
        import qiskit.qasm2
        try:
            custom_env["VIEWER_PRELOADED_QASM"] = qiskit.qasm2.dumps(preloaded_circuit)
        except Exception as e:
            print(f"Warning: Failed to serialize preloaded circuit to QASM. {e}")

    print("Starting local server... A browser window will open automatically once ready.")

    command = [
        sys.executable, "-m", "voila",
        notebook_path,
        "--theme=light",
    ]

    try:
        subprocess.run(command, env=custom_env)
    except KeyboardInterrupt:
        print("\nShutting down Quantum Sandbox...")

def launch_app():
    """
    Launches the master Single Page Application (SPA) in the browser via Voilà.
    """
    print("Initializing Quantum Education Suite SPA...")

    package_dir = os.path.dirname(os.path.abspath(__file__))
    notebook_path = os.path.join(package_dir, "index.ipynb")

    if not os.path.exists(notebook_path):
        print(f"\n❌ ERROR: Could not find 'index.ipynb' at {notebook_path}")
        return

    print("Starting local server... A browser window will open automatically once ready.")

    command = [
        sys.executable, "-m", "voila",
        notebook_path,
        "--theme=light",
    ]

    try:
        # No custom_env needed here, as the SPA handles its own parameters internally
        subprocess.run(command)
    except KeyboardInterrupt:
        print("\nShutting down Quantum Education Suite...")


def launch_challenge(num_qubits=1, initial_state=[1, 0], target_state=[1, -1], show_circuit=True, preloaded_circuit=None):
    """
    Launches the Voilà server with dynamically injected quantum states and an optional solution.
    """
    print(f"Initializing Quantum Challenge Environment ({num_qubits} Qubits)...")

    package_dir = os.path.dirname(os.path.abspath(__file__))
    notebook_path = os.path.join(package_dir, "challenge.ipynb")

    if not os.path.exists(notebook_path):
        print(f"\n❌ ERROR: Could not find 'challenge.ipynb' at {notebook_path}")
        return

    custom_env = os.environ.copy()
    custom_env["CHALLENGE_QUBITS"] = str(num_qubits)
    custom_env["CHALLENGE_INITIAL"] = json.dumps(initial_state)
    custom_env["CHALLENGE_TARGET"] = json.dumps(target_state)
    custom_env["CHALLENGE_SHOW_CIRCUIT"] = "1" if show_circuit else "0"

    # NEW: Serialize the Qiskit circuit to a QASM string
    if preloaded_circuit is not None:
        import qiskit.qasm2
        try:
            custom_env["CHALLENGE_PRELOADED_QASM"] = qiskit.qasm2.dumps(preloaded_circuit)
        except Exception as e:
            print(f"Warning: Failed to serialize preloaded circuit to QASM. {e}")

    print("Starting local server... A browser window will open automatically once ready.")

    command = [
        sys.executable, "-m", "voila",
        notebook_path,
        "--theme=light",
    ]

    try:
        subprocess.run(command, env=custom_env)
    except KeyboardInterrupt:
        print("\nShutting down Quantum Challenge...")

# ==========================================
# SPA ENTRY POINT APPLICATION
# ==========================================

class QuantumViewer:
    """
    The Single Page Application (SPA) entry point.
    """

    def __init__(self):
        self.output = widgets.Output()

        # Load native Qiskit objects directly from memory
        self.algos = QuantumCurriculum.get_algorithms()
        self.challenges = QuantumCurriculum.get_challenges()

        self.title = widgets.HTML("<h1 style='text-align: center; color: #2c3e50;'>Quantum Viewer</h1>")
        self.subtitle = widgets.HTML(
            "<h4 style='text-align: center; color: #7f8c8d; margin-bottom: 20px;'>Select your learning environment</h4>")
        self.header = widgets.VBox([self.title, self.subtitle], layout={'width': '100%'})

        self.tab = widgets.Tab(layout={'width': '500px', 'min_height': '250px'})

        self.tab_sandbox = self._build_sandbox_tab()
        self.tab_algo = self._build_algorithm_tab()
        self.tab_challenge = self._build_challenge_tab()

        self.tab.children = [self.tab_sandbox, self.tab_algo, self.tab_challenge]
        self.tab.titles = ('Sandbox', 'Algorithms', 'Challenges')

        self.menu_container = widgets.VBox(
            [self.header, self.tab],
            layout=widgets.Layout(align_items='center', justify_content='center', width='100%', margin='30px 0px')
        )

    def _build_sandbox_tab(self):
        # Expanded to 9 qubits based on our previous architectural constraints
        self.sb_qubits = widgets.Dropdown(options=[1, 2, 3, 4, 5, 6, 7, 8, 9], value=3, description='Qubits:')
        self.sb_circuit = widgets.Checkbox(value=True, description='Show Circuit UI', indent=False)

        self.sb_initial = widgets.Dropdown(description='Initial State:', layout={'width': '380px'})

        # 1. Initialize the dropdown dynamically based on the default value (3)
        self._update_state_dropdown({'new': self.sb_qubits.value})

        # 2. Bind the observer to mathematically lock state options to qubit dimensions
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

        # Base states available to all dimensions
        options_dict = {
            "|0...0⟩ (Ground State)": "ground",
            "|+...+⟩ (Equal Superposition)": "superposition"
        }

        # Dimension-specific entanglement topologies
        if n == 2:
            options_dict["|Φ⁺⟩ = (|00⟩ + |11⟩)/√2 (Bell State)"] = "ghz"
            options_dict["|Ψ⁺⟩ = (|01⟩ + |10⟩)/√2 (Bell / W-State)"] = "w_state"
        elif n > 2:
            options_dict[f"|GHZ⟩ = (|0...0⟩ + |1...1⟩)/√2 ({n}Q)"] = "ghz"
            options_dict[f"|W⟩ = (|100...⟩ + |010...⟩ + ...)/√{n} ({n}Q)"] = "w_state"

        # Store the current mapping in the class memory for the launch function
        self.sb_states_map = options_dict

        old_val = self.sb_initial.value
        self.sb_initial.options = list(options_dict.keys())

        # Attempt to gracefully retain the user's previous selection if it still exists
        if old_val in self.sb_initial.options:
            self.sb_initial.value = old_val
        else:
            self.sb_initial.value = list(options_dict.keys())[0]

    def _build_algorithm_tab(self):
        options = list(self.algos.keys()) if self.algos else ["No algorithms loaded"]
        self.algo_dropdown = widgets.Dropdown(options=options, description='Algorithm:', layout={'width': '350px'})

        # Define the two configuration toggles
        self.algo_circuit = widgets.Checkbox(value=True, description='Show Circuit UI', indent=False,
                                             layout={'width': '150px'})
        self.algo_final_state = widgets.Checkbox(value=True, description='Show Final State', indent=False,
                                                 layout={'width': '150px'})

        # Group them horizontally to maintain a compact control panel
        checkbox_row = widgets.HBox([self.algo_circuit, self.algo_final_state],
                                    layout={'justify_content': 'center', 'grid_gap': '20px'})

        btn = widgets.Button(description="Study Algorithm", layout={'width': '100%', 'margin': '15px 0px 0px 0px'})
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
        self.chal_circuit = widgets.Checkbox(value=True, description='Show Circuit UI', indent=False)

        btn = widgets.Button(description="Start Challenge", layout={'width': '100%', 'margin': '15px 0px 0px 0px'})
        btn.style.button_color = '#e67e22';
        btn.style.text_color = 'white';
        btn.style.font_weight = 'bold'
        btn.disabled = not bool(self.challenges)
        btn.on_click(self._launch_challenge)
        return widgets.VBox([self.chal_dropdown, self.chal_circuit, btn],
                            layout={'padding': '20px', 'align_items': 'center'})

    def _launch_sandbox(self, b):
        num_qubits = self.sb_qubits.value
        state_key = self.sb_initial.value
        routing_flag = self.sb_states_map[state_key]  # Fetch from the dynamic map

        # We always initialize the physical canvas at the absolute ground state
        # The circuit itself will dictate the final state amplitudes.
        dim = 2 ** num_qubits
        initial_state = [1.0] + [0.0] * (dim - 1)

        preloaded_qc = QuantumCircuit(num_qubits)
        has_circuit = False

        if routing_flag == "superposition":
            # Apply Haddamards to all qubits
            preloaded_qc.h(range(num_qubits))
            has_circuit = True

        elif routing_flag == "ghz":
            # |GHZ>: H on q0, cascaded CNOTs down the register
            preloaded_qc.h(0)
            for i in range(num_qubits - 1):
                preloaded_qc.cx(i, i + 1)
            has_circuit = True

        elif routing_flag == "w_state":
            # |W>: Advanced n-qubit generation using iteratively decaying controlled-Ry rotations
            preloaded_qc.x(0)
            for i in range(num_qubits - 1):
                # Calculate the exact rotation angle to distribute the amplitude evenly
                theta = 2 * np.arccos(1.0 / np.sqrt(num_qubits - i))
                preloaded_qc.cry(theta, i, i + 1)
                preloaded_qc.cx(i + 1, i)
            has_circuit = True

        with self.output:
            clear_output(wait=True)

            if has_circuit:
                # 1. Initialize viewer with ground state and our generated circuit payload
                viewer = InteractiveViewer(num_qubits=num_qubits, initial_state=initial_state,
                                           preloaded_circuit=preloaded_qc, show_circuit=self.sb_circuit.value)

                # 2. Fast-forward the internal state machine so the gates are fully applied upon initialization
                while viewer._redo_circuit_history:
                    viewer._circuit_history.append(viewer.circuit.copy())
                    viewer._action_history.append(viewer._redo_action_history.pop())
                    viewer.circuit = viewer._redo_circuit_history.pop()

                # 3. Mount to the DOM (this renders the final state and the completed circuit image)
                viewer.display()
            else:
                # Ground state requires no circuit generation, render normally
                viewer = InteractiveViewer(num_qubits=num_qubits, initial_state=initial_state,
                                           show_circuit=self.sb_circuit.value)
                viewer.display()

    def _launch_algorithm(self, b):
        qc_raw = self.algos[self.algo_dropdown.value]

        # Isolate initialization instructions to define the true mathematical starting state
        init_circ = QuantumCircuit(qc_raw.num_qubits)
        clean_circ = QuantumCircuit(qc_raw.num_qubits)

        for inst in qc_raw.data:
            if inst.operation.name == 'initialize':
                init_circ.append(inst)
            else:
                clean_circ.append(inst)

        # Calculate the starting tensor from the isolated initialization gates
        initial_state = Statevector.from_instruction(init_circ).data.tolist()

        # The target state must always reflect the full mathematical execution
        target_sv = Statevector.from_instruction(qc_raw).data.tolist()

        with self.output:
            clear_output(wait=True)

            if self.algo_final_state.value:
                # Comparative Mode: Instantiate the dual-canvas ChallengeViewer
                viewer = ChallengeViewer(
                    num_qubits=clean_circ.num_qubits,
                    initial_state=initial_state,
                    target_state=target_sv,
                    preloaded_circuit=clean_circ,
                    show_circuit=self.algo_circuit.value,
                    is_assessment=False
                )
            else:
                # Streamlined Mode: Instantiate the single-canvas InteractiveViewer
                viewer = InteractiveViewer(
                    num_qubits=clean_circ.num_qubits,
                    initial_state=initial_state,
                    preloaded_circuit=clean_circ,
                    show_circuit=self.algo_circuit.value
                )

            viewer.display()

    def _launch_challenge(self, b):
        chal_data = self.challenges[self.chal_dropdown.value]
        with self.output:
            clear_output(wait=True)
            viewer = ChallengeViewer(num_qubits=chal_data["num_qubits"], initial_state=chal_data["initial_state"],
                                     target_state=chal_data["target_state"], preloaded_circuit=None,
                                     show_circuit=self.chal_circuit.value)
            viewer.display()

    def display(self):
        from IPython.display import display as ipy_display, HTML

        # INJECT THE MATHJAX POLYFILL
        # This guarantees ipywidgets will not crash when hunting for the legacy engine.
        ipy_display(HTML("""
        <script>
        window.MathJax = window.MathJax || {};
        window.MathJax.Hub = window.MathJax.Hub || {Queue: function(){}};
        </script>
        """))

        with self.output:
            clear_output(wait=True)
            ipy_display(self.menu_container)
        ipy_display(self.output)