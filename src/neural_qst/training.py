"""Training loops for neural quantum state tomography."""

from __future__ import annotations

from typing import Any, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import vmap
from tqdm.auto import tqdm

from .ansatz import log_derivative
from .gradients import (
    control_variate_coefficient,
    empirical_average_pytree,
    expected_log_derivative_markov,
    stochastic_reconfiguration_gradient,
)
from .metrics import compute_diagnostics
from .sampling import (
    markov_sampling_parallel_log,
    sample_minibatch,
)
from .types import (
    TomographyHistory,
    TomographyResult,
)

Array = Any
PyTree = Any


def simple_gs_tomo(
    H: Optional[Any],
    log_ans: Any,
    params: PyTree,
    n_iter: int,
    batch_size: int,
    freq_data_set: Array,
    freq_space: Array,
    N: int,
    samples_in: int,
    hi: Any,
    learn: float,
    synthe_wave: Optional[Array] = None,
    start_saving: int = 0,
    key: Array = jax.random.PRNGKey(0),
    show_progress: bool = True,
) -> TomographyResult:
    """Run neural-network quantum state tomography."""

    parameters = jax.tree_util.tree_map(
        lambda x: x.copy(),
        params,
    )

    data_probabilities = jnp.asarray(
        freq_data_set,
        dtype=jnp.float32,
    )

    data_probabilities = (
        data_probabilities / jnp.sum(data_probabilities)
    )

    optimizer = optax.sgd(
        learning_rate=learn,
    )

    opt_state = optimizer.init(parameters)

    energies = []
    kls = []
    infidelities = []

    iterator: Sequence[int] = range(n_iter)

    if show_progress:
        iterator = tqdm(iterator)

    for step in iterator:

        key, data_key, model_key = jax.random.split(
            key,
            3,
        )

        minibatch_states, minibatch_counts = (
            sample_minibatch(
                batch_size,
                data_probabilities,
                freq_space,
                data_key,
            )
        )

        minibatch_weights = (
            minibatch_counts / jnp.sum(minibatch_counts)
        )

        if step >= start_saving:

            energy, kl, infidelity = compute_diagnostics(
                H,
                log_ans,
                parameters,
                hi,
                data_probabilities,
                freq_space,
                synthe_wave,
            )

            energies.append(energy)
            kls.append(kl)
            infidelities.append(infidelity)

        model_average = (
            expected_log_derivative_markov(
                log_ans,
                parameters,
                N,
                model_key,
                samples_in,
                markov_sampling_parallel_log,
                hi,
                log_derivative,
            )
        )

        minibatch_derivatives = vmap(
            log_derivative,
            in_axes=(None, None, 0),
        )(
            log_ans,
            parameters,
            minibatch_states,
        )

        gradients = jax.tree_util.tree_map(
            lambda avg, deriv:
            stochastic_reconfiguration_gradient(
                avg,
                deriv,
                minibatch_weights,
            ),
            model_average,
            minibatch_derivatives,
        )

        updates, opt_state = optimizer.update(
            gradients,
            opt_state,
            parameters,
        )

        parameters = optax.apply_updates(
            parameters,
            updates,
        )

    return TomographyResult(
        parameters=parameters,
        history=TomographyHistory(
            energy=np.asarray(energies).real,
            kl_divergence=np.asarray(kls).real,
            infidelity=np.asarray(infidelities).real,
        ),
    )


def simple_gs_tomo_control_variate(
    keyini: Array,
    H: Optional[Any],
    log_ans: Any,
    params: PyTree,
    n_iter: int,
    batch_size: int,
    freq_data_set: Array,
    freq_space: Array,
    N: int,
    samples_in: int,
    hi: Any,
    learn: float,
    synthe_wave: Optional[Array] = None,
    start_save: int = 0,
    control_rate: int = 10,
    mu: float = 0.0,
    show_progress: bool = True,
) -> TomographyResult:
    """Tomography with control variates."""

    parameters = jax.tree_util.tree_map(
        lambda x: x.copy(),
        params,
    )

    data_probabilities = jnp.asarray(
        freq_data_set,
        dtype=jnp.float32,
    )

    data_probabilities = (
        data_probabilities / jnp.sum(data_probabilities)
    )

    key = keyini

    warmup = simple_gs_tomo(
        H=H,
        log_ans=log_ans,
        params=parameters,
        n_iter=1,
        batch_size=batch_size,
        freq_data_set=data_probabilities,
        freq_space=freq_space,
        N=N,
        samples_in=samples_in,
        hi=hi,
        learn=learn,
        synthe_wave=synthe_wave,
        start_saving=10**12,
        key=key,
        show_progress=False,
    )

    previous_parameters = parameters
    parameters = warmup.parameters

    full_previous_derivatives = vmap(
        log_derivative,
        in_axes=(None, None, 0),
    )(
        log_ans,
        previous_parameters,
        freq_space,
    )

    previous_average_data = (
        empirical_average_pytree(
            full_previous_derivatives,
            data_probabilities,
        )
    )

    optimizer = optax.sgd(
        learning_rate=learn,
        momentum=mu,
        nesterov=True,
    )

    opt_state = optimizer.init(parameters)

    energies = []
    kls = []
    infidelities = []

    iterator: Sequence[int] = range(n_iter)

    if show_progress:
        iterator = tqdm(iterator)

    for step in iterator:

        key, data_key, model_key = jax.random.split(
            key,
            3,
        )

        minibatch_states, minibatch_counts = (
            sample_minibatch(
                batch_size,
                data_probabilities,
                freq_space,
                data_key,
            )
        )

        minibatch_weights = (
            minibatch_counts / jnp.sum(minibatch_counts)
        )

        if step >= start_save:

            energy, kl, infidelity = compute_diagnostics(
                H,
                log_ans,
                parameters,
                hi,
                data_probabilities,
                freq_space,
                synthe_wave,
            )

            energies.append(energy)
            kls.append(kl)
            infidelities.append(infidelity)

        current_derivatives = vmap(
            log_derivative,
            in_axes=(None, None, 0),
        )(
            log_ans,
            parameters,
            minibatch_states,
        )

        previous_derivatives = vmap(
            log_derivative,
            in_axes=(None, None, 0),
        )(
            log_ans,
            previous_parameters,
            minibatch_states,
        )

        alpha = control_variate_coefficient(
            current_derivatives,
            previous_derivatives,
            minibatch_weights,
        )

        model_average = (
            expected_log_derivative_markov(
                log_ans,
                parameters,
                N,
                model_key,
                samples_in,
                markov_sampling_parallel_log,
                hi,
                log_derivative,
            )
        )

        gradients = jax.tree_util.tree_map(
            lambda model_avg,
                   current,
                   a,
                   previous,
                   previous_avg:
            model_avg
            - jnp.tensordot(
                minibatch_weights,
                current
                - a * (previous - previous_avg),
                axes=(0, 0),
            ),
            model_average,
            current_derivatives,
            alpha,
            previous_derivatives,
            previous_average_data,
        )

        updates, opt_state = optimizer.update(
            gradients,
            opt_state,
            parameters,
        )

        parameters = optax.apply_updates(
            parameters,
            updates,
        )

        if (
            control_rate > 0
            and (step + 1) % control_rate == 0
        ):

            previous_parameters = (
                jax.tree_util.tree_map(
                    lambda x: x.copy(),
                    parameters,
                )
            )

            full_previous_derivatives = vmap(
                log_derivative,
                in_axes=(None, None, 0),
            )(
                log_ans,
                previous_parameters,
                freq_space,
            )

            previous_average_data = (
                empirical_average_pytree(
                    full_previous_derivatives,
                    data_probabilities,
                )
            )

    return TomographyResult(
        parameters=parameters,
        history=TomographyHistory(
            energy=np.asarray(energies).real,
            kl_divergence=np.asarray(kls).real,
            infidelity=np.asarray(infidelities).real,
        ),
    )