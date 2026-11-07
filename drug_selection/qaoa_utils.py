"""Reusable, ideal-simulator QAOA functions. No parameter-transfer experiments here.

Source of models: Qiskit Optimization Mapper (0.1.x).
Training uses exact StatevectorEstimator expectations; final shots are optional.
"""
from time import perf_counter
import numpy as np
from scipy.optimize import minimize
from qiskit.circuit.library import qaoa_ansatz
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import Statevector
from qiskit import transpile
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
        raise ValueError("Integer positive costs ($M) required; rescale costs/budget if needed")
    if budget < 0 or int(budget) != budget:
        raise ValueError("Budget must be nonnegative integer $M")
    return {
        "data": frame, "names": frame["Drugs"].tolist(),
        "costs": cost.astype(int), "values": value,
        "values_int": np.rint(value * 100).astype(int), "budget": int(budget),
    }


def build_qaoa(instance, layers=1, penalty=None, qubit_limit=20):
    """Knapsack -> penalty QUBO -> Ising -> normalized QAOA circuit."""
    t0 = perf_counter()
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
        raise ValueError(f"{hamiltonian.num_qubits} qubits exceeds local simulation limit {qubit_limit}")
    # Normalization helps keep cost-layer angles numerically manageable.
    # Keep original H and offset for the unscaled physical score.
    nonidentity = [(label, abs(float(coef.real))) for label, coef in zip(
        hamiltonian.paulis.to_labels(), hamiltonian.coeffs
    ) if set(label) != {"I"}]
    scale = max((coeff for _, coeff in nonidentity), default=1.0) or 1.0
    scaled = hamiltonian / scale
    circuit = qaoa_ansatz(cost_operator=scaled, reps=layers, flatten=True)
    return {
        "instance": instance, "application": app, "problem": problem,
        "converter": converter, "qubo": qubo, "hamiltonian": hamiltonian,
        "offset": float(offset), "scale": scale, "scaled_hamiltonian": scaled,
        "circuit": circuit, "parameters": tuple(circuit.parameters),
        "layers": layers, "penalty": converter.penalty,
        "build_seconds": perf_counter() - t0,
    }


def train_qaoa(model, maxiter=100, initial_params=None, seed=42):
    """Optimize expectation with COBYLA; return angles + per-evaluation history."""
    params = model["parameters"]
    if initial_params is None:
        initial_params = np.random.default_rng(seed).uniform(0, np.pi, len(params))
    if len(initial_params) != len(params):
        raise ValueError("Incorrect number of QAOA angles for this layer count")
    estimator = StatevectorEstimator()
    history = []
    def objective(theta):
        bound = model["circuit"].assign_parameters(dict(zip(params, theta)))
        energy = float(estimator.run([(bound, model["scaled_hamiltonian"])]).result()[0].data.evs)
        history.append({"evaluation": len(history) + 1, "energy": energy,
                        "params": np.asarray(theta, dtype=float).copy()})
        return energy
    start = perf_counter()
    result = minimize(objective, np.asarray(initial_params, dtype=float),
                      method="COBYLA", options={"maxiter": maxiter})
    elapsed = perf_counter() - start
    # An interrupted optimization can be evaluated at the best point seen.
    best = min(history, key=lambda h: h["energy"])
    return {
        "params": best["params"], "energy_scaled": best["energy"],
        "evaluations": len(history), "history": history,
        "training_seconds": elapsed, "optimizer_success": bool(result.success),
    }


def decoded_distribution(model, params):
    """Exact distribution of ORIGINAL drug selections; sums over slack assignments."""
    start = perf_counter()
    circuit = model["circuit"].assign_parameters(dict(zip(model["parameters"], params)))
    prob = Statevector.from_instruction(circuit).probabilities()
    instance = model["instance"]
    n = len(instance["costs"])
    # Converter decodes the QUBO's auxiliary variables; never assume slack layout.
    totals = {}
    for index, p in enumerate(prob):
        if p <= 0:
            continue
        bits = [(index >> j) & 1 for j in range(model["hamiltonian"].num_qubits)]
        original = model["converter"].interpret(bits)
        selected = tuple(model["application"].interpret(np.asarray(original)))
        if any(i < 0 or i >= n for i in selected):
            raise ValueError("Converter returned an invalid drug index")
        totals[selected] = totals.get(selected, 0.0) + float(p)
    return totals, perf_counter() - start


def shots_for_confidence(success_probability, confidence=0.95):
    """Independent draws needed to see a target solution at least once."""
    q = float(success_probability)
    if q <= 0: return np.inf
    if q >= 1: return 1
    return int(np.ceil(np.log1p(-confidence) / np.log1p(-q)))


def evaluate_qaoa(model, params, optimum, target_fraction=0.95, shots=0, seed=42):
    """Evaluate fixed angles WITHOUT training; ideal probabilities, optional sampled shots."""
    dist, evaluation_seconds = decoded_distribution(model, params)
    instance = model["instance"]
    costs, values, budget = instance["costs"], instance["values"], instance["budget"]
    def value(sel): return float(sum(values[i] for i in sel))
    def feasible(sel): return sum(int(costs[i]) for i in sel) <= budget
    target = target_fraction * optimum
    q_feasible = sum(p for sel,p in dist.items() if feasible(sel))
    q_target = sum(p for sel,p in dist.items() if feasible(sel) and value(sel) + 1e-8 >= target)
    q_optimal = sum(p for sel,p in dist.items() if feasible(sel) and abs(value(sel)-optimum) <= 0.0051)
    # This 'best' summarizes the ideal distribution support, not the finite-shot outcome.
    valid = [sel for sel,p in dist.items() if p > 1e-12 and feasible(sel)]
    best_ideal = max(valid, key=value) if valid else None
    sampled_best = None
    if shots:
        rng = np.random.default_rng(seed)
        keys = list(dist)
        draw = rng.choice(len(keys), size=shots, p=np.asarray(list(dist.values())) / sum(dist.values()))
        seen = [keys[int(i)] for i in draw if feasible(keys[int(i)])]
        sampled_best = max(seen, key=value) if seen else None
    return {
        "prob_feasible": q_feasible, "prob_target": q_target, "prob_optimal": q_optimal,
        "shots_95": shots_for_confidence(q_target),
        "best_ideal": best_ideal, "best_sampled": sampled_best,
        "best_sampled_value": value(sampled_best) if sampled_best is not None else np.nan,
        "distribution": dist, "evaluation_seconds": evaluation_seconds,
        "final_shots": int(shots),
    }


def circuit_resources(model, topology="line", optimization_level=1, seed=42):
    """Native-basis counts on a SYNTHETIC coupling map, not actual QPU timings."""
    circuit = model["circuit"]
    n = circuit.num_qubits
    basis = ["rz", "sx", "x", "cx"]
    kwargs = {"basis_gates": basis, "optimization_level": optimization_level,
              "seed_transpiler": seed}
    start = perf_counter()
    logical = transpile(circuit, **kwargs)
    if topology == "line":
        coupling = CouplingMap.from_line(n, bidirectional=True)
    elif topology == "all_to_all":
        coupling = CouplingMap.from_full(n)
    else:
        raise ValueError("topology must be 'line' or 'all_to_all'")
    mapped = transpile(circuit, coupling_map=coupling, **kwargs)
    two_qubit = sum(inst.operation.num_qubits == 2 for inst in mapped.data)
    return {
        "qubits": n, "logical_cx": logical.count_ops().get("cx", 0),
        "transpiled_two_qubit": two_qubit, "transpiled_depth": mapped.depth(),
        "topology": topology, "transpile_seconds": perf_counter() - start,
        "native_ops": dict(mapped.count_ops()),
    }
