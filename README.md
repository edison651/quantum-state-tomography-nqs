# Neural Quantum State Tomography with JAX and NetKet

Neural-network quantum state tomography (QST) for pure quantum states using neural quantum states, JAX, and NetKet.

This repository implements tomography from measurements in a single computational basis using stochastic optimization and Markov-chain sampling. The implementation was originally developed for reconstructing ground states of one-dimensional Ising-type systems.

## Features

- Neural quantum state ansatz
- JAX-based automatic differentiation
- NetKet Metropolis sampling
- Mini-batch stochastic optimization
- Optional control-variate variance reduction
- Fidelity and KL-divergence diagnostics

## Installation

```bash
git clone https://github.com/edison651/qst_master.git
cd neural-qst

pip install -e .
```

## Dependencies

- Python >= 3.10
- jax
- netket
- optax
- numpy
- tqdm

## Example

```python
from neural_qst import simple_gs_tomo

result = simple_gs_tomo(
    H=H,
    log_ans=model,
    params=params,
    n_iter=2000,
    batch_size=512,
    freq_data_set=data_probabilities,
    freq_space=data_states,
    N=N,
    samples_in=500,
    hi=hilbert,
    learn=1e-3,
)
```

The returned object contains:
- optimized parameters,
- energy history,
- KL divergence history,
- reconstruction infidelity.

## Repository Structure

```text
src/neural_qst/
├── ansatz.py
├── gradients.py
├── metrics.py
├── sampling.py
├── training.py
├── types.py
└── __init__.py
```

## Notes

The ansatz is assumed to return logarithmic amplitudes:

\[
\log \psi_\\theta(x)
=
\log \langle x | \psi_\\theta \\rangle
\]

This repository is intended primarily as a research and educational implementation rather than a production tomography framework.