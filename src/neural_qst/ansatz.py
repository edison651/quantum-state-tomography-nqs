"""Ansatz utilities for neural quantum state tomography."""

from __future__ import annotations

from typing import Any

import jax

Array = Any
PyTree = Any


def log_psi(ansatz: Any, parameters: PyTree, basis_state: Array) -> Array:
    """Evaluate the log-amplitude log(<x|psi_theta>)."""
    return ansatz.apply(parameters, basis_state)


log_derivative = jax.grad(log_psi, argnums=1)

# Backwards-compatible alias
op_O_k = log_derivative