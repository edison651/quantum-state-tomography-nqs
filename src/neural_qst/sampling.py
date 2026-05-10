"""Sampling utilities for neural quantum state tomography."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import netket as nk
from jax import jit

Array = Any
PyTree = Any

@jit
def _sample_indices_from_distribution(probabilities: Array, uniforms: Array) -> Array:
    """Sample indices from a normalized discrete probability distribution."""
    cdf = jnp.cumsum(probabilities)
    return jnp.searchsorted(cdf, uniforms, side="left")


def sample_minibatch(
    n_samples: int,
    probabilities: Array,
    basis_states: Array,
    key: Array,
) -> tuple[Array, Array]:
    """Draw a minibatch from an empirical basis distribution.

    Returns the unique sampled states and their counts.
    """
    probabilities = jnp.asarray(probabilities, dtype=jnp.float32)
    probabilities = probabilities / jnp.sum(probabilities)

    uniforms = jax.random.uniform(key, shape=(n_samples,))
    indices = _sample_indices_from_distribution(probabilities, uniforms)
    sampled_states = basis_states[indices]

    return jnp.unique(sampled_states, axis=0, return_counts=True)


def markov_sampling_parallel_log(
    n_samples: int,
    log_ansatz: Any,
    parameters: PyTree,
    n_sites: int,
    key: Array,
    hilbert: Any,
    return_counts: bool = True,
    n_chains: int = 10,
    n_sweeps: int = 10,
) -> tuple[Array, Array] | Array:
    """Sample configurations from |psi|^2 using NetKet Metropolis sampling."""
    rule = nk.sampler.rules.LocalRule()
    sampler = nk.sampler.MetropolisSampler(
        hilbert,
        rule,
        n_chains=n_chains,
        n_sweeps=n_sweeps,
    )

    sampler_state = sampler.init_state(log_ansatz, parameters, key)
    sampler_state = sampler.reset(log_ansatz, parameters, sampler_state)

    samples, _ = sampler.sample(
        log_ansatz,
        parameters,
        state=sampler_state,
        chain_length=n_samples,
    )

    samples = samples.reshape(-1, n_sites)

    if return_counts:
        return jnp.unique(samples, axis=0, return_counts=True)

    return samples


# Optional backwards-compatible alias
give_me_batches = sample_minibatch