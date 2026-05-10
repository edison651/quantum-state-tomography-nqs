"""Shared types and result containers for neural quantum state tomography."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

PyTree = Any


@dataclass
class TomographyHistory:
    """Training diagnostics collected during tomography."""

    energy: np.ndarray
    kl_divergence: np.ndarray
    infidelity: np.ndarray


@dataclass
class TomographyResult:
    """Output of a tomography optimization run."""

    parameters: PyTree
    history: TomographyHistory