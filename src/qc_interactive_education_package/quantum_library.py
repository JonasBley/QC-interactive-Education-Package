import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import QFTGate, grover_operator
from qiskit.quantum_info import Statevector, random_statevector

class QuantumCurriculum:
    """
    A native Python constructor for the quantum education suite.
    Bypasses serialization to preserve modern Qiskit architectures,
    including advanced phase gates and custom instructions.
    """

    @staticmethod
    def get_algorithms():
        algos = {}

        # --- Algorithm 1: Bell State ---
        qc_bell = QuantumCircuit(2)
        qc_bell.h(0)
        qc_bell.cx(0, 1)
        algos["Bell State Entanglement"] = qc_bell

        # --- Algorithm 2: GHZ State ---
        qc_ghz = QuantumCircuit(3)
        qc_ghz.h(0)
        qc_ghz.cx(0, 1)
        qc_ghz.cx(1, 2)
        algos["GHZ State Entanglement"] = qc_ghz

        # --- Algorithm 9: Quantum Teleportation (Randomized) ---
        qc_teleport = QuantumCircuit(3)

        # 1. Generate a mathematically random pure state for Alice to teleport
        # Seed is omitted to ensure a uniquely random statevector on every launch
        rand_psi = random_statevector(2)
        rand_psi = Statevector(np.exp(-1j * np.angle(rand_psi.data[0])) * rand_psi.data).data
        qc_teleport.initialize(rand_psi, 0)

        # 2. Establish the EPR pair (Entanglement) between Alice (q1) and Bob (q2)
        qc_teleport.h(1)
        qc_teleport.cx(1, 2)

        # 3. Alice performs a Bell-basis transformation on her local qubits (q0, q1)
        qc_teleport.cx(0, 1)
        qc_teleport.h(0)

        # 4. Bob applies conditional Pauli corrections via the Deferred Measurement Principle
        # If q1 is |1>, apply X. If q0 is |1>, apply Z.
        qc_teleport.cx(1, 2)
        qc_teleport.cz(0, 2)

        algos["Quantum Teleportation (Random State)"] = qc_teleport

        # --- Algorithm 3: Quantum Fourier Transform ---
        qc_qft = QuantumCircuit(3)
        qc_qft.append(QFTGate(3), [0, 1, 2])
        algos["Quantum Fourier Transform (3Q)"] = qc_qft.decompose()

        # --- Algorithm 4: Grover's Search (|101>) ---
        qc_grover = QuantumCircuit(3)
        qc_grover.h([0, 1, 2])
        oracle = QuantumCircuit(3)
        oracle.cz(0, 2)
        qc_grover = qc_grover.compose(grover_operator(oracle))
        algos["Grover's Search (Target |101>)"] = qc_grover

        # --- Algorithm 5: 3-Qubit Bit-Flip Error Correction ---
        qc_3q_err = QuantumCircuit(3)
        # 1. Encode logical |1>
        qc_3q_err.x(0)
        qc_3q_err.cx(0, 1)
        qc_3q_err.cx(0, 2)
        # 2. Simulate environment error (Bit flip on q0)
        qc_3q_err.x(0)
        # 3. Syndrome measurement and correction via Toffoli
        qc_3q_err.cx(0, 1)
        qc_3q_err.cx(0, 2)
        qc_3q_err.ccx(1, 2, 0)
        algos["Error Correction: 3-Qubit Bit-Flip"] = qc_3q_err

        # --- Algorithm 6: 7-Qubit Steane Code (Logical |0>) ---
        qc_steane = QuantumCircuit(7)
        # Data qubits in superposition
        qc_steane.h([0, 1, 2])
        # Hamming code parity check entanglement
        qc_steane.cx(0, 3)
        qc_steane.cx(1, 3)
        qc_steane.cx(0, 4)
        qc_steane.cx(2, 4)
        qc_steane.cx(1, 5)
        qc_steane.cx(2, 5)
        qc_steane.cx(0, 6)
        qc_steane.cx(1, 6)
        qc_steane.cx(2, 6)
        algos["Error Correction: Steane [[7,1,3]] Code"] = qc_steane

        # --- Algorithm 7: Shor's Period Finding (a=2, N=3) ---
        qc_shor = QuantumCircuit(4)
        # Initialize counting register
        qc_shor.h([0, 1])
        # Initialize work register to |1> (binary 01)
        qc_shor.x(3)
        # Modular Exponentiation: 2^x mod 3
        # If counting bit 0 is 1, multiply by 2 mod 3 (swaps 01 and 10)
        qc_shor.cswap(0, 2, 3)
        # (If counting bit 1 is 1, multiply by 4 mod 3. Since 4 = 1 mod 3, this is an Identity operation)
        # Inverse QFT on the counting register
        qc_shor.swap(0, 1)
        qc_shor.h(1)
        qc_shor.cp(-np.pi / 2, 0, 1)
        qc_shor.h(0)
        algos["Shor's Algorithm: Period Finding"] = qc_shor

        # --- Algorithm 8: Full Shor's Algorithm (8Q, N=15, a=7) ---
        qc_shor_8q = QuantumCircuit(8)

        # 1. Initialize counting register (Qubits 0-3) in full superposition
        qc_shor_8q.h([0, 1, 2, 3])

        # 2. Initialize work register (Qubits 4-7) to the eigenstate |1> (binary 0001)
        qc_shor_8q.x(4)

        # 3. Define the mathematical Oracle for modular exponentiation
        def c_amod15(a, power):
            """Generates a controlled unitary for a^power mod 15"""
            U = QuantumCircuit(4)
            for _ in range(power):
                if a in [7, 8]:
                    U.swap(2, 3)
                    U.swap(1, 2)
                    U.swap(0, 1)
                if a in [7, 11, 13]:
                    for q in range(4):
                        U.x(q)
            U = U.to_gate()
            U.name = f"{a}^{power} mod 15"
            return U.control(1)

        # 4. Apply controlled modular exponentiations across the counting register
        qc_shor_8q.append(c_amod15(7, 1), [0, 4, 5, 6, 7])
        qc_shor_8q.append(c_amod15(7, 2), [1, 4, 5, 6, 7])
        qc_shor_8q.append(c_amod15(7, 4), [2, 4, 5, 6, 7])
        qc_shor_8q.append(c_amod15(7, 8), [3, 4, 5, 6, 7])

        # 5. Apply the Inverse Quantum Fourier Transform to extract the phase
        qc_shor_8q.append(QFTGate(4).inverse(), [0, 1, 2, 3])

        algos["Shor's Algorithm: Factor 15 (8Q)"] = qc_shor_8q

        return algos

    @staticmethod
    def get_challenges():
        challenges = {}
        inv_sq2 = 1.0 / np.sqrt(2)

        challenges["Level 1: Create a Superposition (|+⟩)"] = {
            "num_qubits": 1,
            "initial_state": [1.0, 0.0],
            "target_state": [inv_sq2, inv_sq2]
        }

        challenges["Level 2: Phase Flip (|1⟩ to |-⟩)"] = {
            "num_qubits": 1,
            "initial_state": [0.0, 1.0],
            "target_state": [inv_sq2, -inv_sq2]
        }

        qc_bell = QuantumCircuit(2)
        qc_bell.h(0)
        qc_bell.cx(0, 1)
        challenges["Level 3: Construct a Bell State"] = {
            "num_qubits": 2,
            "initial_state": [1.0, 0.0, 0.0, 0.0],
            "target_state": Statevector.from_instruction(qc_bell).data.tolist()
        }

        qc_ghz = QuantumCircuit(3)
        qc_ghz.h(0)
        qc_ghz.cx(0, 1)
        qc_ghz.cx(1, 2)
        challenges["Level 4: Construct a GHZ state"] = {
            "num_qubits": 3,
            "initial_state": [1.0] + [0.0] * 7,
            "target_state": Statevector.from_instruction(qc_ghz).data.tolist()
        }

        challenges["Level 5: Quantum Teleportation"] = {
            "num_qubits": 3,
            "initial_state": [inv_sq2, -inv_sq2, 0.0, 0.0, 0.0, 0.0, inv_sq2, -inv_sq2],
            "target_state": [inv_sq2, -inv_sq2, inv_sq2, -inv_sq2, -inv_sq2, inv_sq2, inv_sq2, -inv_sq2]
        }

        challenges["Level 6: Search Challenge: Amplify |101⟩"] = {
            "num_qubits": 3,
            "initial_state": [1.0] + [0.0] * 7,
            "target_state": [0.0, 0.0, 0.0, 0.0, 0.0, inv_sq2, 0.0, inv_sq2]
        }

        # --- Phase-Flip Error Correction ---
        # 1. Define the pristine target: Logical |1> encoded into the phase basis (|--->)
        qc_phase_target = QuantumCircuit(3)
        qc_phase_target.x(0)
        qc_phase_target.cx(0, 1)
        qc_phase_target.cx(0, 2)
        qc_phase_target.h([0, 1, 2])

        # 2. Define the corrupted initial state: A Z-error strikes Qubit 1 (index 0)
        qc_phase_init = qc_phase_target.copy()
        qc_phase_init.z(0)

        challenges["Level 7: Correct a Phase-Flip Error"] = {
            "num_qubits": 3,
            "initial_state": Statevector.from_instruction(qc_phase_init).data.tolist(),
            "target_state": Statevector.from_instruction(qc_phase_target).data.tolist()
        }

        return challenges