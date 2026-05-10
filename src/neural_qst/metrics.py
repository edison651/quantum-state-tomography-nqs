"""Observables and diagnostic metrics for neural QST."""

from __future__ import annotations

from typing import Any, Optional

import jax.numpy as jnp

Array = Any
PyTree = Any

_EPS = 1e-12


def expectation_value(
    operator: Any,
    parameters: PyTree,
    ansatz: Any,
    basis_states: Array,
    return_state: bool = False,
):
    """Compute <psi|O|psi>/<psi|psi>."""

    connected_states, matrix_elements = operator.get_conn_padded(
        basis_states
    )

    log_psi_x = ansatz.apply(parameters, basis_states)
    psi_x = jnp.exp(log_psi_x)

    norm = jnp.sum(jnp.abs(psi_x) ** 2)

    log_psi_y = ansatz.apply(parameters, connected_states)

    local_terms = matrix_elements * jnp.exp(
        jnp.conjugate(log_psi_x[:, None]) + log_psi_y
    )

    value = jnp.sum(local_terms) / norm

    if return_state:
        normalized_psi = psi_x / jnp.sqrt(norm)
        return value, normalized_psi

    return value


def safe_kl_divergence(
    data_probabilities: Array,
    model_probabilities: Array,
) -> Array:
    """Compute KL(data || model) with numerical protection."""

    data_probabilities = (
        data_probabilities / jnp.sum(data_probabilities)
    )

    model_probabilities = (
        model_probabilities / jnp.sum(model_probabilities)
    )

    return jnp.sum(
        jnp.where(
            data_probabilities > 0,
            data_probabilities
            * (
                jnp.log(data_probabilities + _EPS)
                - jnp.log(model_probabilities + _EPS)
            ),
            0.0,
        )
    )


def infidelity(
    target_state: Array,
    reconstructed_state: Array,
) -> Array:
    """Compute pure-state infidelity."""

    target_state = target_state / jnp.sqrt(
        jnp.sum(jnp.abs(target_state) ** 2)
    )

    reconstructed_state = reconstructed_state / jnp.sqrt(
        jnp.sum(jnp.abs(reconstructed_state) ** 2)
    )

    overlap = jnp.vdot(target_state, reconstructed_state)

    return 1.0 - jnp.abs(overlap) ** 2


def compute_diagnostics(
    hamiltonian: Optional[Any],
    ansatz: Any,
    parameters: PyTree,
    hilbert: Any,
    data_probabilities: Array,
    data_states: Array,
    target_state: Optional[Array] = None,
):
    """Compute energy, KL divergence, and infidelity."""

    all_states = hilbert.all_states()

    state_numbers = hilbert.states_to_numbers(
        data_states
    )

    if hamiltonian is None:

        log_values = ansatz.apply(
            parameters,
            all_states,
        )

        psi = jnp.exp(log_values)

        psi = psi / jnp.sqrt(
            jnp.sum(jnp.abs(psi) ** 2)
        )

        energy = jnp.nan + 0.0j

    else:

        energy, psi = expectation_value(
            hamiltonian,
            parameters,
            ansatz,
            all_states,
            return_state=True,
        )

    model_probabilities = jnp.abs(
        psi[state_numbers]
    ) ** 2

    kl = safe_kl_divergence(
        data_probabilities,
        model_probabilities,
    )

    if target_state is None:
        infid = jnp.nan

    else:
        infid = infidelity(
            target_state,
            psi,
        )

    return (
        complex(energy),
        float(jnp.real(kl)),
        float(jnp.real(infid)),
    )


# Backwards-compatible alias
expect_op = expectation_value