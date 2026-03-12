import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, random_statevector


class QuantumCurriculum:
    """
    A native Python constructor for the Quantum Viewer.
    """

    @staticmethod
    def annotate(qc, text, step_index=None):
        """
        Attaches a pedagogical annotation to a specific mathematical step in the timeline.
        If step_index is None, it dynamically binds to the current length of the circuit,
        meaning the annotation will appear exactly as the *next* gate is drawn.
        """
        if step_index is None:
            step_index = len(qc.data)

        if qc.metadata is None:
            qc.metadata = {}
        if 'annotations' not in qc.metadata:
            qc.metadata['annotations'] = {}

        qc.metadata['annotations'][step_index] = text
        return qc

    @staticmethod
    def get_algorithms():
        algos = {}

        # ==========================================
        # ALGORITHM 1: Bell State
        # ==========================================
        qc_bell = QuantumCircuit(2)
        QuantumCurriculum.annotate(qc_bell,
                                   "We start in the absolute ground state $|00\\rangle$. Both qubits possess deterministic, classical values.")

        qc_bell.h(0)
        QuantumCurriculum.annotate(qc_bell,
                                   "The Hadamard gate on $q_0$ creates a perfect superposition. The system is now separated into $\\frac{|00\\rangle + |01\\rangle}{\\sqrt{2}}$.")

        qc_bell.cx(0, 1)
        QuantumCurriculum.annotate(qc_bell,
                                   "The CNOT$_{01}$ gate permanently entangles the qubits. Notice how the amplitudes have shifted along the axis of Qubit 1 into the mathematically inseparable Bell State: $| \\Phi^+ \\rangle = \\frac{|00\\rangle + |11\\rangle}{\\sqrt{2}}$.")

        algos["Bell State Entanglement"] = qc_bell

        # ==========================================
        # ALGORITHM 1.5: GHZ State
        # ==========================================
        qc_ghz = QuantumCircuit(3)
        QuantumCurriculum.annotate(qc_ghz, "We begin in the $|000\\rangle$ ground state.")

        qc_ghz.h(0)
        QuantumCurriculum.annotate(qc_ghz,
                                   "The **Hadamard** gate places qubit 0 into a superposition, yielding $\\frac{|000\\rangle + |001\\rangle}{\\sqrt{2}}$.")

        qc_ghz.cx(0, 1)
        QuantumCurriculum.annotate(qc_ghz,
                                   "The first **CNOT** entangles $q_0$ and $q_1$. We now have a bipartite entangled pair tensored with a deterministic $q_2$.")

        qc_ghz.cx(1, 2)
        QuantumCurriculum.annotate(qc_ghz,
                                   "The second **CNOT** cascades the entanglement to $q_2$. We have successfully generated the maximally entangled tripartite **GHZ State**: $\\frac{|000\\rangle + |111\\rangle}{\\sqrt{2}}$.")

        algos["GHZ State Entanglement"] = qc_ghz

        # ==========================================
        # ALGORITHM 2: Unitary Quantum Teleportation
        # ==========================================
        qc_teleport = QuantumCircuit(3)

        psi = random_statevector(2)
        psi = Statevector(np.exp(-1j * np.angle(psi.data[0])) * psi.data)
        full_sv = Statevector.from_label('00').tensor(psi)
        qc_teleport.initialize(full_sv, [0, 1, 2])
        QuantumCurriculum.annotate(qc_teleport,
                                   "**Initialization:** Qubit 0 holds the unknown (here, random) state $|\\psi\\rangle$. Qubits 1 and 2 act as the blank computational medium.")

        # Step 1: Create Entanglement
        qc_teleport.h(1)
        QuantumCurriculum.annotate(qc_teleport,
                                   "**Entanglement Generation:** A Hadamard and CNOT gate construct the maximally entangled Bell state $|\\Phi^+\\rangle$ between Qubits 1 and 2.")
        qc_teleport.cx(1, 2)
        qc_teleport.barrier()

        # Step 2: Alice's local operations
        qc_teleport.cx(0, 1)
        QuantumCurriculum.annotate(qc_teleport,
                                   "**Basis Transformation:** Alice interacts her message qubit (Qubit 0) with her half of the entangled pair (Qubit 1) using a CNOT followed by a Hadamard. This operation effectively projects the system into the Bell basis. Notice how the information previously stored in qubit 0 is now stored in qubit 2, while the last operations only acted along the axis of qubit 0 and 1! This is the crux of the teleportation algorithm and only works because of the previous entanglement between qubits 1 and 2.")
        qc_teleport.h(0)
        qc_teleport.barrier()

        # Step 3: Quantum Feedforward
        qc_teleport.cx(1, 2)
        QuantumCurriculum.annotate(qc_teleport,
                                   "**Quantum Feedforward:** To satisfy the requirements of a unitary Statevector simulator, we utilize the *Deferred Measurement Principle*. Instead of extracting classical bits, Alice's qubits act as direct quantum controls for the corrective Pauli operations on Bob's qubit. The exact state $|\\psi\\rangle$ is successfully reconstructed on Qubit 2, successfully disentangling the system.")
        qc_teleport.cz(0, 2)

        algos["Quantum Teleportation"] = qc_teleport

        # ==========================================
        # ALGORITHM 3: DEUTSCH-JOZSA (Balanced, n=3)
        # ==========================================
        qc_dj = QuantumCircuit(4)
        QuantumCurriculum.annotate(qc_dj,
                                   r"We begin in the absolute ground state $|0000\rangle$. Qubits 0-2 form the $n=3$ input register, and Qubit 3 is the auxiliary (ancilla) evaluation qubit.")

        qc_dj.x(3)
        QuantumCurriculum.annotate(qc_dj,
                                   r"We apply an **X-gate** to the ancilla ($q_3$) to prepare it in the $|1\rangle$ state. This is a strict mathematical prerequisite for the phase kickback mechanism.")

        # Bind annotation right as the superposition begins
        QuantumCurriculum.annotate(qc_dj,
                                   r"We apply **Hadamard** gates to all qubits. The input register is now in a uniform superposition of all $2^n$ basis states, and the ancilla is in the negative phase state $|-\rangle$.",
                                   step_index=len(qc_dj.data) + 1)
        for q in range(4):
            qc_dj.h(q)

        qc_dj.cx(0, 3)
        QuantumCurriculum.annotate(qc_dj,
                                   r"We apply the **Balanced Oracle**, here: cascading CX gates from each input to the ancilla (you can see it via the 'unpack custom gates' option at the top). By entangling the inputs to the $|-\rangle$ ancilla via **CNOT** gates, the function's output is written directly into the phase of the input superposition via phase kickback! The global state vectors are now mathematically shifted.")
        qc_dj.cx(1, 3)
        qc_dj.cx(2, 3)



        QuantumCurriculum.annotate(qc_dj,
                                   r"We apply final **Hadamard** gates to the input register to force interference. Because exactly half the phases were flipped by the balanced oracle, destructive interference mathematically eliminates the $|000\rangle$ amplitude. Any measurement will yield a non-zero state, definitively proving the function is balanced.",
                                   step_index=len(qc_dj.data) + 1)
        for q in range(3):
            qc_dj.h(q)

        algos["Deutsch-Jozsa Algorithm: Balanced (3Q Input)"] = qc_dj

        # ==========================================
        # ALGORITHM 4: GROVER'S SEARCH DYNAMIC BUILDER
        # ==========================================
        def build_grover(target_bitstring):
            n = len(target_bitstring)
            qc = QuantumCircuit(n)
            QuantumCurriculum.annotate(qc, "Grover's Search begins in the absolute ground state $|0...0\\rangle$.")

            QuantumCurriculum.annotate(qc,
                                       "We apply **Hadamard** gates to all qubits, spreading the amplitude evenly across all $2^n$ basis states to create a uniform superposition.",
                                       step_index=1)
            qc.h(range(n))

            oracle = QuantumCircuit(n, name="Oracle")
            for i, bit in enumerate(reversed(target_bitstring)):
                if bit == '0': oracle.x(i)
            oracle.h(n - 1)
            oracle.mcx(list(range(n - 1)), n - 1)
            oracle.h(n - 1)
            for i, bit in enumerate(reversed(target_bitstring)):
                if bit == '0': oracle.x(i)
            oracle_gate = oracle.to_gate(label="Oracle")

            diffuser = QuantumCircuit(n, name="Diffuser")
            diffuser.h(range(n))
            diffuser.x(range(n))
            diffuser.h(n - 1)
            diffuser.mcx(list(range(n - 1)), n - 1)
            diffuser.h(n - 1)
            diffuser.x(range(n))
            diffuser.h(range(n))
            diffuser_gate = diffuser.to_gate(label="Diffuser")

            optimal_iterations = int(np.floor((np.pi / 4.0) * np.sqrt(2 ** n)))

            for i in range(optimal_iterations):
                qc.append(oracle_gate, range(n))
                QuantumCurriculum.annotate(qc,
                                           f"Iteration {i + 1}: The **Oracle** isolates the target string $|{target_bitstring}\\rangle$ and applies a geometric reflection, flipping its phase (amplitude) to negative.")

                qc.append(diffuser_gate, range(n))
                QuantumCurriculum.annotate(qc,
                                           f"Iteration {i + 1}: The **Diffuser** performs an inversion about the mean. Notice how this geometrically drains the amplitude from the uniform states and physically amplifies the target state.")

            return qc

        algos["Grover's Search: Target |1011⟩ (4Q)"] = build_grover("1011")
        algos["Grover's Search: Target |10101⟩ (5Q)"] = build_grover("10101")

        # ==========================================
        # ALGORITHM 5: QUANTUM FOURIER TRANSFORM
        # ==========================================
        qc_qft = QuantumCircuit(3)
        QuantumCurriculum.annotate(qc_qft,
                                   "We initialize the Quantum Fourier Transform (QFT). The QFT maps the computational basis into the Fourier (phase) basis.")

        qc_qft.h(2)
        QuantumCurriculum.annotate(qc_qft,
                                   "A **Hadamard** on the Most Significant Bit (MSB, $q_2$) begins the phase fractionalization.")

        qc_qft.cp(np.pi / 2, 1, 2)
        QuantumCurriculum.annotate(qc_qft,
                                   r"A Controlled-Phase gate ($\pi/2$) applies a fractional kickback conditional on $q_1$.")

        qc_qft.cp(np.pi / 4, 0, 2)
        QuantumCurriculum.annotate(qc_qft,
                                   r"A Controlled-Phase gate ($\pi/4$) applies a finer fractional kickback conditional on $q_0$. The MSB is now fully encoded in the Fourier basis.")

        qc_qft.h(1)
        qc_qft.cp(np.pi / 2, 0, 1)
        qc_qft.h(0)

        QuantumCurriculum.annotate(qc_qft,
                                   "Finally, the **SWAP** gate reverses the qubit ordering to mathematically align the output with standard Qiskit little-endian notation.",
                                   step_index=len(qc_qft.data) + 1)
        qc_qft.swap(0, 2)

        algos["Quantum Fourier Transform (3Q)"] = qc_qft

        # ==========================================
        # ALGORITHM 6: SHOR'S PERIOD FINDING (a=2, N=3)
        # ==========================================
        qc_shor = QuantumCircuit(4)
        QuantumCurriculum.annotate(qc_shor,
                                   "We initialize a 4-qubit Shor's algorithm. Qubits 0-1 serve as the counting register, and Qubits 2-3 are the auxiliary work register.")

        QuantumCurriculum.annotate(qc_shor,
                                   "We prepare the counting register in a uniform superposition and excite the auxiliary register to $|1\\rangle$ (the multiplicative identity).",
                                   step_index=1)
        qc_shor.h([0, 1])
        qc_shor.x(3)

        qc_shor.cswap(0, 2, 3)
        qc_shor.swap(0, 1)
        qc_shor.h(1)
        qc_shor.cp(-1 * np.pi / 2, 0, 1)
        qc_shor.h(0)
        QuantumCurriculum.annotate(qc_shor,
                                   "After executing the modular exponentiation and the Inverse QFT, the counting register collapses into the periodic phase shift, revealing the mathematical period of the function.")

        algos["Shor's Algorithm: Period Finding"] = qc_shor

        # ==========================================
        # ALGORITHM 7: SHOR'S PERIOD FINDING (6Q, a=5, N=6)
        # ==========================================
        qc_shor_6q = QuantumCircuit(6)
        n_count = 3

        QuantumCurriculum.annotate(qc_shor_6q,
                                   "We initialize the counting register in a uniform superposition and set the auxiliary work register to the multiplicative identity $|1\\rangle$ (Decimal 1). In the following, modular exponentiation will be computed in superposition, forcing a phase kickback that effectively encodes the arithmetic periodicity of the function directly into the amplitudes of the counting register.",
                                   step_index=1)
        qc_shor_6q.h([0, 1, 2])
        qc_shor_6q.x(3)
        qc_shor_6q.barrier()

        QuantumCurriculum.annotate(qc_shor_6q,
                                   r"We apply $5^1 \\bmod 6$ controlled by $q_0$. Because the work register starts at $|1\rangle$, multiplying by 5 toggles the state strictly between $|1\rangle$ (001) and $|5\rangle$ (101). This mathematical elegance allows us to execute the exact unitary operation using a single Controlled-NOT gate on $q_5$, completely avoiding black-box matrices.",
                                   step_index=len(qc_shor_6q.data) + 1)
        qc_shor_6q.cx(0, 5)
        qc_shor_6q.barrier()

        QuantumCurriculum.annotate(qc_shor_6q,
                                   r"We conceptually apply $5^2 \\bmod 6$ and $5^4 \\bmod 6$ controlled by $q_1$ and $q_2$. However, because $5^2 = 25 \equiv 1 \bmod 6$, both of these operations evaluate to the Identity matrix. The early saturation of these higher-order exponentiations explicitly proves the period is $r=2$. No physical gates are required.")
        qc_shor_6q.barrier()

        QuantumCurriculum.annotate(qc_shor_6q,
                                   "The explicit **Inverse Quantum Fourier Transform** perfectly resolves the accumulated kickback phases, collapsing the counting register exclusively into the fraction periods $0$ and $4$. ($4/8 = 1/2$, yielding period $r=2$).",
                                   step_index=len(qc_shor_6q.data) + 1)
        for qubit in range(n_count // 2):
            qc_shor_6q.swap(qubit, n_count - qubit - 1)
        for j in range(n_count):
            for m in range(j):
                qc_shor_6q.cp(-np.pi / float(2 ** (j - m)), m, j)
            qc_shor_6q.h(j)

        algos["Shor's Algorithm: Period Finding (a=5, N=6)"] = qc_shor_6q

        # ==========================================
        # SHOR'S ALGORITHM (8Q) - SEQUENTIAL EXECUTION
        # ==========================================
        qc_shor_8q = QuantumCircuit(8)
        n_count = 4
        a = 7

        prep = QuantumCircuit(8, name="Initialize")
        for q in range(n_count):
            prep.h(q)
        prep.x(n_count)

        qc_shor_8q.append(prep.to_gate(label="Initialization"), range(8))
        QuantumCurriculum.annotate(qc_shor_8q,
                                   r"We initialize the 8Q Shor algorithm. The 4-qubit counting register is placed into uniform superposition, and the auxiliary register is excited to $|1\rangle$.")

        QuantumCurriculum.annotate(qc_shor_8q,
                                   r"The following sequential modular exponentiation computes $f(x) = a^x \bmod N$ in superposition. This operation imprints the hidden periodic structure of the function onto the counting register's phase via phase kickback, creating a highly complex entanglement topology between the two registers.", step_index=len(qc_shor_8q.data) + 1)
        def c_amod15(base, power):
            if base not in [2, 4, 7, 8, 11, 13]:
                raise ValueError("'base' must be 2,4,7,8,11 or 13")
            U = QuantumCircuit(4)
            for _ in range(power):
                if base in [2, 13]:
                    U.swap(0, 1);
                    U.swap(1, 2);
                    U.swap(2, 3)
                if base in [7, 8]:
                    U.swap(2, 3);
                    U.swap(1, 2);
                    U.swap(0, 1)
                if base in [4, 11]:
                    U.swap(1, 3);
                    U.swap(0, 2)
                if base in [7, 11, 13]:
                    for q in range(4):
                        U.x(q)
            U_gate = U.to_gate()
            U_gate.name = f"{base}^{power} mod 15"
            return U_gate.control(1)

        for q in range(n_count):
            qc_shor_8q.append(c_amod15(a, 2 ** q), [q] + [i + n_count for i in range(4)])

        QuantumCurriculum.annotate(qc_shor_8q,
                                   r"We have completed the sequential modular exponentiation. Notice the highly complex entanglement topology mapping between the counting and work registers.")

        def qft_dagger(n):
            qc = QuantumCircuit(n, name="QFT†")
            for qubit in range(n // 2):
                qc.swap(qubit, n - qubit - 1)
            for j in range(n):
                for m in range(j):
                    qc.cp(-np.pi / float(2 ** (j - m)), m, j)
                qc.h(j)
            return qc.to_gate(label="Inverse QFT")

        qc_shor_8q.append(qft_dagger(n_count), range(n_count))
        QuantumCurriculum.annotate(qc_shor_8q,
                                   r"The **Inverse QFT** isolates the periodic interference patterns. Measuring the counting register now yields one of the exact eigenvalues required to extract the period classically.")

        algos["Shor's Algorithm: Factor 15 (8Q)"] = qc_shor_8q

        # ==========================================
        # ALGORITHM 9: 3-Qubit Bit-Flip Error Correction
        # ==========================================
        qc_3q_err = QuantumCircuit(3)
        QuantumCurriculum.annotate(qc_3q_err,
                                   "We start in the $|000\\rangle$ ground state. The objective is to encode a single logical qubit across three physical qubits to protect against a bit-flip error.")

        qc_3q_err.x(0)
        QuantumCurriculum.annotate(qc_3q_err,
                                   "We initialize $q_0$ to the state $|1\\rangle$. This is the raw logical state we wish to protect.")

        QuantumCurriculum.annotate(qc_3q_err,
                                   "Encoding complete. By cascading CNOT gates, we have mapped the logical state into the physical repetition code $|111\\rangle$.",
                                   step_index=len(qc_3q_err.data) + 1)
        qc_3q_err.cx(0, 1)
        qc_3q_err.cx(0, 2)

        qc_3q_err.x(0)
        QuantumCurriculum.annotate(qc_3q_err,
                                   "⚠️ **ERROR INJECTED:** A quantum noise event (X-gate) strikes $q_0$, flipping it back to $|0\\rangle$. The system is now corrupted into the state $|011\\rangle$.")

        QuantumCurriculum.annotate(qc_3q_err,
                                   "Syndrome Measurement: We compute the parity of the qubits using CNOTs. This maps the error syndrome into the ancilla space without collapsing the logical superposition.",
                                   step_index=len(qc_3q_err.data) + 1)
        qc_3q_err.cx(0, 1)
        qc_3q_err.cx(0, 2)

        qc_3q_err.ccx(1, 2, 0)
        QuantumCurriculum.annotate(qc_3q_err,
                                   "Correction: The Toffoli (CCX) gate acts as an autonomous classical logic switch, flipping $q_0$ back to its correct state exclusively if the syndrome flags an error. The state is restored to $|111\\rangle$.")

        algos["Error Correction: 3-Qubit Bit-Flip"] = qc_3q_err

        # ==========================================
        # ALGORITHM 10: 7-Qubit Steane Code
        # ==========================================
        qc_steane = QuantumCircuit(7)
        QuantumCurriculum.annotate(qc_steane,
                                   "We initialize the Steane [[7,1,3]] Error Correction code. This CSS code can simultaneously correct both bit-flip (X) and phase-flip (Z) errors on any single qubit.")

        QuantumCurriculum.annotate(qc_steane,
                                   "Hadamard gates prepare the three data qubits into a uniform superposition.",
                                   step_index=1)
        qc_steane.h([0, 1, 2])

        QuantumCurriculum.annotate(qc_steane,
                                   "A highly specific array of parity operations encodes the logical $|0\\rangle_L$ state into the 7 physical qubits using the classical Hamming code topology.",
                                   step_index=len(qc_steane.data) + 1)
        qc_steane.cx(0, 3);
        qc_steane.cx(1, 3)
        qc_steane.cx(0, 4);
        qc_steane.cx(2, 4)
        qc_steane.cx(1, 5);
        qc_steane.cx(2, 5)
        qc_steane.cx(0, 6);
        qc_steane.cx(1, 6);
        qc_steane.cx(2, 6)
        qc_steane.barrier()

        QuantumCurriculum.annotate(qc_steane,
                                   "⚠️ **ERROR INJECTED:** A severe noise event strikes $q_0$, causing both a bit-flip (X) and a phase-flip (Z). This is mathematically equivalent to a lethal Y-error. The logical state is now severely corrupted.",
                                   step_index=len(qc_steane.data) + 1)
        qc_steane.x(0)
        qc_steane.z(0)
        qc_steane.barrier()

        QuantumCurriculum.annotate(qc_steane,
                                   "To isolate the error without using auxiliary syndrome qubits, we apply the exact inverse of the encoding circuit. If no error had occurred, this would return the system flawlessly to $|0000000\\rangle$.",
                                   step_index=len(qc_steane.data) + 1)
        qc_steane.cx(2, 6);
        qc_steane.cx(1, 6);
        qc_steane.cx(0, 6)
        qc_steane.cx(2, 5);
        qc_steane.cx(1, 5)
        qc_steane.cx(2, 4);
        qc_steane.cx(0, 4)
        qc_steane.cx(1, 3);
        qc_steane.cx(0, 3)
        qc_steane.h([0, 1, 2])
        qc_steane.barrier()

        QuantumCurriculum.annotate(qc_steane,
                                   "**Correction:** The Y-error has deterministically collapsed into a unique geometric footprint, altering the state to $-|1001101\\rangle$. We apply corrective X-gates to clear the physical bit-flips on qubits 0, 3, 4, and 6, and a Z-gate to clear the negative phase.",
                                   step_index=len(qc_steane.data) + 1)
        qc_steane.z(0)
        qc_steane.x([0, 3, 4, 6])
        qc_steane.barrier()

        QuantumCurriculum.annotate(qc_steane,
                                   "With the error mathematically neutralized and the vacuum state restored, we run the encoding array a second time. The logical $|0\\rangle_L$ state is flawlessly reconstructed, surviving a multi-axis strike.",
                                   step_index=len(qc_steane.data) + 1)
        qc_steane.h([0, 1, 2])
        qc_steane.cx(0, 3);
        qc_steane.cx(1, 3)
        qc_steane.cx(0, 4);
        qc_steane.cx(2, 4)
        qc_steane.cx(1, 5);
        qc_steane.cx(2, 5)
        qc_steane.cx(0, 6);
        qc_steane.cx(1, 6);
        qc_steane.cx(2, 6)

        algos["Error Correction: Steane [[7,1,3]] Code"] = qc_steane

        return algos

    @staticmethod
    def get_challenges():
        challenges = {}
        inv_sq2 = 1.0 / np.sqrt(2)

        def make_hint(num_qubits, text):
            qc_hint = QuantumCircuit(num_qubits)
            QuantumCurriculum.annotate(qc_hint, text, step_index=0)
            return qc_hint

        challenges["Level 1: Create a Superposition (|+⟩)"] = {
            "num_qubits": 1,
            "initial_state": [1.0, 0.0],
            "target_state": [inv_sq2, inv_sq2],
            "preloaded_circuit": make_hint(1,
                                           r"To map a deterministic classical state into a uniform superposition, apply a **Hadamard** (`H`) gate."),
            "available_gates": ['H', 'X', 'Y', 'Z'],
            "max_gate_count": 1
        }

        challenges["Level 2: Phase Flip (|1⟩ to |-⟩)"] = {
            "num_qubits": 1,
            "initial_state": [0.0, 1.0],
            "target_state": [inv_sq2, -inv_sq2],
            "preloaded_circuit": make_hint(1,
                                           r"You are starting in the $|1\rangle$ state. Applying a Hadamard gate to $|1\rangle$ will yield the targeted negative phase superposition automatically."),
            "available_gates": ['H', 'X', 'Y', 'Z'],
            "max_gate_count": 1
        }

        qc_bell = QuantumCircuit(2)
        qc_bell.h(0);
        qc_bell.cx(0, 1)
        challenges["Level 3: Construct a Bell State"] = {
            "num_qubits": 2,
            "initial_state": [1.0, 0.0, 0.0, 0.0],
            "target_state": Statevector.from_instruction(qc_bell).data.tolist(),
            "preloaded_circuit": make_hint(2,
                                           r"Entanglement requires two steps: First, create a superposition on $q_0$ using an `H` gate. Then, correlate $q_1$ to $q_0$ using a Controlled-NOT (`CX`) gate."),
            "available_gates": ['H', 'X', 'Y', 'Z'],
            "max_gate_count": 2
        }

        qc_ghz = QuantumCircuit(3)
        qc_ghz.h(0);
        qc_ghz.cx(0, 1);
        qc_ghz.cx(1, 2)
        challenges["Level 4: Construct a GHZ state"] = {
            "num_qubits": 3,
            "initial_state": [1.0] + [0.0] * 7,
            "target_state": Statevector.from_instruction(qc_ghz),
            "preloaded_circuit": make_hint(3,
                                           r"To build a tripartite GHZ state, first construct a standard Bell State between $q_0$ and $q_1$, and then cascade a second `CX` gate from $q_1$ to $q_2$."),
            "available_gates": ['H', 'X', 'Y', 'Z'],
            "max_gate_count": 3
        }

        psi = random_statevector(2)
        psi = Statevector(np.exp(-1j * np.angle(psi.data[0])) * psi.data)
        full_sv = Statevector.from_label('00').tensor(psi)
        qc_teleport = QuantumCircuit(3)
        qc_teleport.initialize(full_sv, [0, 1, 2])
        qc_teleport.h(1);
        qc_teleport.cx(1, 2)
        qc_teleport.cx(0, 1);
        qc_teleport.h(0)

        challenges["Level 5: Quantum Teleportation"] = {
            "num_qubits": 3,
            "initial_state": full_sv,
            "target_state": Statevector.from_instruction(qc_teleport),
            "preloaded_circuit": make_hint(3,
                                           r"You must bind Bob (qubit 3) to Alice (qubit 2) using a Bell State, then mathematically fuse Alice's random state (qubit 1) into the Bell pair using `CX` and `H` gates. The measurement, in the end, would then yield a state that can be correct to the initial state of qubit 1."),
            "available_gates": ['H', 'X', 'Y', 'Z'],
            "max_gate_count": 4
        }

        challenges["Level 6: Search Challenge: Amplify |101⟩"] = {
            "num_qubits": 3,
            "initial_state": [1.0] + [0.0] * 7,
            "target_state": [0.0, 0.0, 0.0, 0.0, 0.0, inv_sq2, 0.0, inv_sq2],
            "preloaded_circuit": make_hint(3,
                                           r"To amplify $|101\rangle$, you must construct an Oracle that isolates the target using $X$-gates on the zero-bits, and phase-flips it using a Multi-Controlled Z-gate. (For a 3-qubit challenge, standard `H` gates and `CCX` work perfectly)."),
            "available_gates": ['H', 'X', 'Y', 'Z'],
            "max_gate_count": 26
        }

        qc_phase_target = QuantumCircuit(3)
        qc_phase_target.x(0)
        qc_phase_target.cx(0, 1);
        qc_phase_target.cx(0, 2)
        qc_phase_target.h([0, 1, 2])

        qc_phase_init = qc_phase_target.copy()
        qc_phase_init.z(0)

        challenges["Level 7: Correct a Phase-Flip Error"] = {
            "num_qubits": 3,
            "initial_state": Statevector.from_instruction(qc_phase_init).data.tolist(),
            "target_state": Statevector.from_instruction(qc_phase_target).data.tolist(),
            "preloaded_circuit": make_hint(3,
                                           r"The system is currently corrupted by a $Z$-error. Because Phase ($Z$) errors become Bit ($X$) errors when rotated into the Hadamard basis, you must sandwich your standard Repetition Code syndrome measurement (`CX`, `CCX`) between two arrays of `H` gates."),
            "available_gates": ['H', 'X', 'Y', 'Z'],
            "max_gate_count": 9
        }

        return challenges