"""Small, controlled knapsack perturbations and phase-scale transfer utilities.

No full-state introspection. No classical optimum is used when choosing parameters.
"""

import numpy as np
import pandas as pd

from qaoa_utils import make_instance


def transfer_cases(dataset, source, include_larger=False):
    """Return labelled variations. Unchanged source drugs keep their row order."""
    base = source["data"].copy()
    budget = source["budget"]
    num_drugs = len(base)
    if len(dataset) < 2 * num_drugs + 2:
        raise ValueError("Dataset needs at least twice the base roster size plus two rows")
    value_column = "Expected value ($M)"
    cost_column = "Test Cost ($M)"

    # Change one factor per case so its effect can be compared with the control.
    cases = [("unchanged/control", base.copy(), budget)]

    changed = base.copy()
    changed.loc[0, value_column] *= 1.05
    cases.append(("single value +5%", changed, budget))

    changed = base.copy()
    changed.loc[0, cost_column] += 1
    cases.append(("single cost +$1M", changed, budget))

    changed = base.copy()
    changed.loc[0, value_column] *= 1.05
    changed.loc[1, value_column] *= 0.95
    cases.append(("two values +/-5%", changed, budget))

    changed = base.copy()
    changed.loc[0, cost_column] += 1
    changed.loc[1, cost_column] = max(1, int(changed.loc[1, cost_column]) - 1)
    cases.append(("two costs +/-$1M", changed, budget))

    cases.extend([
        ("budget -$2M", base.copy(), max(0, budget - 2)),
        ("budget +$2M", base.copy(), budget + 2),
    ])

    changed = base.copy()
    changed.loc[:, value_column] *= 1.05
    cases.append(("all values +5%", changed, budget))

    one_new_drug = dataset.iloc[[num_drugs]].copy().reset_index(drop=True)
    replacement_roster = pd.concat([base.iloc[:-1], one_new_drug], ignore_index=True)
    expanded_roster = pd.concat([base, one_new_drug], ignore_index=True)
    cases.extend([
        ("replace one drug", replacement_roster, budget),
        ("different same-size roster", dataset.iloc[num_drugs:2 * num_drugs].copy(), budget),
        ("add one drug", expanded_roster, budget),
    ])
    if include_larger:
        two_new_drugs = dataset.iloc[num_drugs:num_drugs + 2]
        larger_roster = pd.concat([base, two_new_drugs], ignore_index=True)
        independent_roster = dataset.iloc[num_drugs:2 * num_drugs + 2].copy()
        cases.extend([
            ("add two drugs", larger_roster, budget),
            ("larger independent roster", independent_roster, budget),
        ])

    return [
        (name, make_instance(frame, case_budget))
        for name, frame, case_budget in cases
    ]


def coefficient_gamma_transfer(source_model, target_model, params, threshold=0.2):
    """Phase-match gamma for approximately proportional normalized Hamiltonians.

    If H_target ≈ ratio * H_source (up to identity), divide gamma by ratio.
    This is exact only for proportional Hamiltonians; otherwise it is a heuristic.
    Skip automatically if qubit widths differ or fit residual exceeds threshold.
    """
    params = np.asarray(params, dtype=float)
    if source_model["hamiltonian"].num_qubits != target_model["hamiltonian"].num_qubits:
        return params.copy(), {
            "applied": False,
            "reason": "qubit-count mismatch",
            "ratio": np.nan,
            "residual": np.nan,
        }

    def nonidentity_coefficients(operator):
        # Identity terms add a global phase, so they do not affect angle transfer.
        return {
            label: float(coefficient.real)
            for label, coefficient in zip(operator.paulis.to_labels(), operator.coeffs)
            if set(label) != {"I"}
        }

    source_coefficients = nonidentity_coefficients(source_model["scaled_hamiltonian"])
    target_coefficients = nonidentity_coefficients(target_model["scaled_hamiltonian"])
    labels = sorted(set(source_coefficients) | set(target_coefficients))
    source_vector = np.array([source_coefficients.get(label, 0.0) for label in labels])
    target_vector = np.array([target_coefficients.get(label, 0.0) for label in labels])

    # Fit target ≈ ratio * source using the same Pauli-term order in both vectors.
    source_norm_squared = source_vector @ source_vector
    ratio = (
        float(source_vector @ target_vector / source_norm_squared)
        if source_norm_squared > 0 else np.nan
    )
    fit_error = np.linalg.norm(target_vector - ratio * source_vector)
    residual = float(fit_error / max(np.linalg.norm(target_vector), 1e-12))
    if not np.isfinite(ratio) or ratio <= 0 or residual > threshold:
        return params.copy(), {
            "applied": False,
            "reason": "Hamiltonians not close to proportional",
            "ratio": ratio,
            "residual": residual,
        }

    result = params.copy()
    for index, parameter in enumerate(target_model["parameters"]):
        if parameter.name.startswith("γ") or parameter.name.lower().startswith("gamma"):
            result[index] /= ratio
    return result, {
        "applied": True,
        "reason": "approximate coefficient scaling",
        "ratio": ratio,
        "residual": residual,
    }
