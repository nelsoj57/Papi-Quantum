"""Reusable, ideal-simulator QAOA functions. No parameter-transfer experiments here.

Source of models: Qiskit Optimization Mapper (0.1.x).
Training supports exact expectations or finite-shot energy estimates.
"""

from time import perf_counter

import numpy as np
from scipy.optimize import minimize

from qiskit import transpile
from qiskit.circuit.library import qaoa_ansatz
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import Statevector
from qiskit.transpiler import CouplingMap
from qiskit_addon_opt_mapper.applications import Knapsack
from qiskit_addon_opt_mapper.converters import OptimizationProblemToQubo
from qiskit_addon_opt_mapper.translators import to_ising


def make_instance(frame, budget):
    """Preserve all CSV columns, using integer $M costs and centi-$M values."""
    frame = frame.copy().reset_index(drop=True)
    frame.columns = frame.columns.str.strip()
    cost = frame["Test Cost ($M)"].astype(float).to_numpy()
    value = frame["Expected value ($M)"].astype(float).to_numpy()
    if not len(cost) or not np.all(np.isfinite(cost)) or not np.all(np.isfinite(value)):
        raise ValueError("The selected CSV rows must have finite costs and values")
    if np.any(cost <= 0) or not np.allclose(cost, np.rint(cost)):
        raise ValueError(
            "Integer positive costs ($M) required; rescale costs/budget if needed"
        )
    if budget < 0 or int(budget) != budget:
        raise ValueError("Budget must be nonnegative integer $M")
    return {
        "data": frame,
        "names": frame["Drugs"].tolist(),
        "costs": cost.astype(int),
        "values": value,
        "values_int": np.rint(value * 100).astype(int),
        "budget": int(budget),
    }


def build_qaoa(instance, layers=1, penalty=None, qubit_limit=20):
    """Knapsack -> penalty QUBO -> Ising -> normalized QAOA circuit."""
    start = perf_counter()
    app = Knapsack(
        values=instance["values_int"].tolist(),
        weights=instance["costs"].tolist(),
        max_weight=instance["budget"],
    )
    problem = app.to_optimization_problem()
    converter = OptimizationProblemToQubo(penalty=penalty)
    qubo = converter.convert(problem)
    hamiltonian, offset = to_ising(qubo)
    if hamiltonian.num_qubits > qubit_limit:
        raise ValueError(
            f"{hamiltonian.num_qubits} qubits exceeds local simulation limit {qubit_limit}"
        )
    # Normalization helps keep cost-layer angles numerically manageable.
    # Keep original H and offset for the unscaled physical score.
    nonidentity_magnitudes = [
        abs(float(coefficient.real))
        for label, coefficient in zip(hamiltonian.paulis.to_labels(), hamiltonian.coeffs)
        if set(label) != {"I"}
    ]
    scale = max(nonidentity_magnitudes, default=1.0) or 1.0
    scaled = hamiltonian / scale
    circuit = qaoa_ansatz(cost_operator=scaled, reps=layers, flatten=True)
    return {
        "instance": instance,
        "application": app,
        "problem": problem,
        "converter": converter,
        "qubo": qubo,
        "hamiltonian": hamiltonian,
        "offset": float(offset),
        "scale": scale,
        "scaled_hamiltonian": scaled,
        "circuit": circuit,
        "parameters": tuple(circuit.parameters),
        "layers": layers,
        "penalty": converter.penalty,
        "build_seconds": perf_counter() - start,
    }


def train_qaoa(model, maxiter=100, initial_params=None, seed=42, shots_per_eval=None):
    """COBYLA with exact energy (default) or hardware-compatible finite-shot estimates."""
    params = model["parameters"]
    if initial_params is None:
        initial_params = np.random.default_rng(seed).uniform(0, np.pi, len(params))
    if len(initial_params) != len(params):
        raise ValueError("Incorrect number of QAOA angles for this layer count")
    if shots_per_eval is not None and int(shots_per_eval) <= 0:
        raise ValueError("shots_per_eval must be a positive integer")
    estimator = StatevectorEstimator() if shots_per_eval is None else None
    history = []

    def objective(theta):
        bound = model["circuit"].assign_parameters(dict(zip(params, theta)))
        if shots_per_eval is None:
            job = estimator.run([(bound, model["scaled_hamiltonian"])])
            energy = float(job.result()[0].data.evs)
        else:
            # Statevector is only an execution backend here: the optimizer sees
            # finite-shot bitstrings, never exact amplitudes or probabilities.
            state = Statevector.from_instruction(bound)
            state.seed(seed + len(history))
            shot_count = int(shots_per_eval)
            counts = state.sample_counts(shots=shot_count)
            total_energy = sum(
                int(count) * _ising_eigenvalue(model["scaled_hamiltonian"], bits)
                for bits, count in counts.items()
            )
            energy = total_energy / shot_count
        history.append({
            "evaluation": len(history) + 1,
            "energy": energy,
            "params": np.asarray(theta, dtype=float).copy(),
        })
        return energy

    start = perf_counter()
    result = minimize(
        objective,
        np.asarray(initial_params, dtype=float),
        method="COBYLA",
        options={"maxiter": maxiter},
    )
    elapsed = perf_counter() - start
    # COBYLA's final point need not be the best point it evaluated.
    best = min(history, key=lambda entry: entry["energy"])
    training_shots = 0 if shots_per_eval is None else len(history) * int(shots_per_eval)
    return {
        "params": best["params"],
        "energy_scaled": best["energy"],
        "evaluations": len(history),
        "history": history,
        "training_seconds": elapsed,
        "optimizer_success": bool(result.success),
        "training_shots": training_shots,
        "shots_per_eval": shots_per_eval,
    }


def decoded_distribution(model, params):
    """Exact distribution of ORIGINAL drug selections; sums over slack assignments."""
    start = perf_counter()
    circuit = model["circuit"].assign_parameters(dict(zip(model["parameters"], params)))
    probabilities = Statevector.from_instruction(circuit).probabilities()
    instance = model["instance"]
    num_drugs = len(instance["costs"])
    # Converter decodes the QUBO's auxiliary variables; never assume slack layout.
    totals = {}
    for index, probability in enumerate(probabilities):
        if probability <= 0:
            continue
        # Qiskit puts qubit 0 at the right of a displayed bitstring.
        bits = [
            (index >> qubit) & 1
            for qubit in range(model["hamiltonian"].num_qubits)
        ]
        original = model["converter"].interpret(bits)
        selected = tuple(model["application"].interpret(np.asarray(original)))
        if any(index < 0 or index >= num_drugs for index in selected):
            raise ValueError("Converter returned an invalid drug index")
        totals[selected] = totals.get(selected, 0.0) + float(probability)
    return totals, perf_counter() - start


def shots_for_confidence(success_probability, confidence=0.95):
    """Independent draws needed to see a target solution at least once."""
    probability = float(success_probability)
    if probability <= 0:
        return np.inf
    if probability >= 1:
        return 1
    # Solve 1 - (1 - probability)**shots >= confidence for shots.
    return int(np.ceil(np.log1p(-confidence) / np.log1p(-probability)))


def evaluate_qaoa(model, params, optimum, target_fraction=0.95, shots=0, seed=42):
    """Evaluate fixed angles WITHOUT training; ideal probabilities, optional sampled shots."""
    distribution, evaluation_seconds = decoded_distribution(model, params)
    instance = model["instance"]
    costs = instance["costs"]
    values = instance["values"]
    budget = instance["budget"]

    def selection_value(selected):
        return float(sum(values[index] for index in selected))

    def is_feasible(selected):
        return sum(int(costs[index]) for index in selected) <= budget

    target = target_fraction * optimum
    probability_feasible = 0.0
    probability_target = 0.0
    probability_optimal = 0.0
    valid_selections = []
    for selected, probability in distribution.items():
        if not is_feasible(selected):
            continue
        value = selection_value(selected)
        probability_feasible += probability
        if value + 1e-8 >= target:
            probability_target += probability
        # Keep the existing half-centi-$M tolerance for identifying optimal values.
        if abs(value - optimum) <= 0.0051:
            probability_optimal += probability
        if probability > 1e-12:
            valid_selections.append(selected)

    # The ideal best describes the distribution, rather than an observed sample.
    best_ideal = max(valid_selections, key=selection_value) if valid_selections else None
    sampled_best = None
    if shots:
        rng = np.random.default_rng(seed)
        selections = list(distribution)
        probabilities = np.asarray(list(distribution.values()))
        probabilities = probabilities / sum(distribution.values())
        draws = rng.choice(len(selections), size=shots, p=probabilities)
        feasible_draws = [
            selections[int(index)]
            for index in draws
            if is_feasible(selections[int(index)])
        ]
        sampled_best = max(feasible_draws, key=selection_value) if feasible_draws else None
    return {
        "prob_feasible": probability_feasible,
        "prob_target": probability_target,
        "prob_optimal": probability_optimal,
        "shots_95": shots_for_confidence(probability_target),
        "best_ideal": best_ideal,
        "best_sampled": sampled_best,
        "best_sampled_value": (
            selection_value(sampled_best) if sampled_best is not None else np.nan
        ),
        "distribution": distribution,
        "evaluation_seconds": evaluation_seconds,
        "final_shots": int(shots),
    }


def circuit_resources(model, topology="line", optimization_level=1, seed=42):
    """Native-basis counts on a SYNTHETIC coupling map, not actual QPU timings."""
    circuit = model["circuit"]
    num_qubits = circuit.num_qubits
    basis = ["rz", "sx", "x", "cx"]
    transpile_options = {
        "basis_gates": basis,
        "optimization_level": optimization_level,
        "seed_transpiler": seed,
    }
    start = perf_counter()
    logical = transpile(circuit, **transpile_options)
    if topology == "line":
        coupling = CouplingMap.from_line(num_qubits, bidirectional=True)
    elif topology == "all_to_all":
        coupling = CouplingMap.from_full(num_qubits)
    else:
        raise ValueError("topology must be 'line' or 'all_to_all'")
    mapped = transpile(circuit, coupling_map=coupling, **transpile_options)
    two_qubit = sum(inst.operation.num_qubits == 2 for inst in mapped.data)
    return {
        "qubits": num_qubits,
        "logical_cx": logical.count_ops().get("cx", 0),
        "transpiled_two_qubit": two_qubit,
        "transpiled_depth": mapped.depth(),
        "topology": topology,
        "transpile_seconds": perf_counter() - start,
        "native_ops": dict(mapped.count_ops()),
    }


def _ising_eigenvalue(operator, bits):
    """Cost of a measured computational-basis string for a diagonal Ising operator."""
    value = int(bits.replace(" ", ""), 2)
    result = 0.0
    for pauli, coefficient in zip(operator.paulis.to_labels(), operator.coeffs):
        if set(pauli) - {"I", "Z"}:
            raise ValueError("Expected a diagonal Ising cost operator")
        # A Z term contributes -1 for an odd number of measured 1s on its qubits.
        mask = int("".join("1" if char == "Z" else "0" for char in pauli), 2)
        parity = (value & mask).bit_count() % 2
        eigenvalue = -1 if parity else 1
        result += float(coefficient.real) * eigenvalue
    return result


def sample_qaoa(model, params, shots, optimum=None, target_fraction=0.95, seed=42):
    """QPU-compatible evaluation using only measured bitstrings, never exact amplitudes.

    Statevector simulates the circuit locally but outputs only finite-shot counts.
    An actual QPU Sampler can replace this execution backend later.
    """
    if shots < 1:
        raise ValueError("At least one final shot is required")
    start = perf_counter()
    bound = model["circuit"].assign_parameters(dict(zip(model["parameters"], params)))
    state = Statevector.from_instruction(bound)
    state.seed(seed)
    counts = state.sample_counts(shots=shots)
    instance = model["instance"]
    costs = instance["costs"]
    values = instance["values"]
    budget = instance["budget"]
    successes = 0
    optimal_hits = 0
    feasible_shots = 0
    best_selection = None
    best_value = float("-inf")
    rows = []
    for bitstring, count in counts.items():
        # The converter expects qubit 0 first, opposite the displayed bitstring.
        all_bits = [int(bit) for bit in bitstring[::-1]]
        decoded = model["converter"].interpret(all_bits)
        drug_indices = model["application"].interpret(np.asarray(decoded))
        selected = tuple(sorted(int(index) for index in drug_indices))
        cost = sum(int(costs[i]) for i in selected)
        value = sum(float(values[i]) for i in selected)
        feasible = cost <= budget
        if feasible:
            feasible_shots += int(count)
            if value > best_value:
                best_selection = selected
                best_value = value
        meets_target = (
            feasible
            and optimum is not None
            and value + 1e-8 >= target_fraction * optimum
        )
        if meets_target:
            successes += int(count)
        if feasible and optimum is not None and abs(value - optimum) <= 0.0051:
            optimal_hits += int(count)
        rows.append({
            "selection": selected,
            "count": int(count),
            "value": value,
            "cost": cost,
            "feasible": feasible,
            "target": meets_target,
        })
    return {
        "best_sampled": best_selection,
        "best_sampled_value": best_value if best_selection is not None else np.nan,
        "prob_target_sampled": successes / shots if optimum is not None else np.nan,
        "prob_optimal_sampled": optimal_hits / shots if optimum is not None else np.nan,
        "prob_feasible_sampled": feasible_shots / shots,
        "target_hits": successes,
        "optimal_hits": optimal_hits,
        "shots": int(shots),
        "outcomes": rows,
        "sampling_seconds": perf_counter() - start,
    }
