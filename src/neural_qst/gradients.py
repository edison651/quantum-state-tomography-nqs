"""Gradient estimators and control variates for neural QST."""

from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as jnp
from jax import jit, vmap

from .ansatz import log_derivative

Array = Any
PyTree = Any

_EPS = 1e-12


def empirical_average_pytree(values: PyTree, weights: Array) -> PyTree:
    """Compute weighted averages over the first axis of a PyTree."""
    weights = weights / jnp.sum(weights)

    return jax.tree_util.tree_map(
        lambda x: jnp.tensordot(weights, x, axes=(0, 0)),
        values,
    )


def expected_log_derivative_markov(
    ansatz: Any,
    parameters: PyTree,
    n_sites: int,
    key: Array,
    n_markov_samples: int,
    sampler_fn: Callable,
    hilbert: Any,
    derivative_fn: Callable = log_derivative,
) -> PyTree:
    """Estimate <O_lambda>_{p_lambda} using Markov-chain sampling."""
    states, counts = sampler_fn(
        n_markov_samples,
        ansatz,
        parameters,
        n_sites,
        key,
        hilbert,
    )

    weights = counts / jnp.sum(counts)

    derivatives = vmap(
        derivative_fn,
        in_axes=(None, None, 0),
    )(ansatz, parameters, states)

    return empirical_average_pytree(derivatives, weights)


@jit
def stochastic_reconfiguration_gradient(
    model_average: Array,
    minibatch_derivatives: Array,
    minibatch_weights: Array,
) -> Array:
    """Gradient estimator for single-basis tomography."""
    minibatch_weights = minibatch_weights / jnp.sum(minibatch_weights)

    data_average = jnp.tensordot(
        minibatch_weights,
        minibatch_derivatives,
        axes=(0, 0),
    )

    return jnp.asarray(
        model_average - data_average,
        dtype=jnp.complex64,
    )


def control_variate_coefficient(
    current_derivatives: PyTree,
    previous_derivatives: PyTree,
    weights: Array,
) -> PyTree:
    """Compute the optimal linear control-variate coefficient."""
    weights = weights / jnp.sum(weights)

    @jit
    def _coefficient(
        current: Array,
        previous: Array,
        weights_: Array,
    ) -> Array:

        mean_current = jnp.tensordot(weights_, current, axes=(0, 0))
        mean_previous = jnp.tensordot(weights_, previous, axes=(0, 0))

        centered_current = current - mean_current
        centered_previous = previous - mean_previous

        covariance = jnp.tensordot(
            weights_,
            centered_current * jnp.conjugate(centered_previous),
            axes=(0, 0),
        )

        variance = jnp.tensordot(
            weights_,
            jnp.abs(centered_previous) ** 2,
            axes=(0, 0),
        )

        return jnp.nan_to_num(covariance / (variance + _EPS))

    return jax.tree_util.tree_map(
        lambda current, previous: _coefficient(
            current,
            previous,
            weights,
        ),
        current_derivatives,
        previous_derivatives,
    )


# Backwards-compatible aliases
expected_D_lambda_markov = expected_log_derivative_markov
SGD = stochastic_reconfiguration_gradient
finding_cv = control_variate_coefficient