"""
D-Wave Ocean SDK integration (or stub).

If the `dimod` package is available (part of D-Wave Ocean SDK), this solver
uses the real SimulatedAnnealingSampler from dimod and optionally the QPU
via the dwave-system package.  If neither is installed, it falls back to
our own SA solver and logs a clear warning.

Usage
-----
    solver = DWaveStubSolver(use_qpu=False)   # local SA via dimod or built-in
    solver = DWaveStubSolver(use_qpu=True)    # attempts real QPU (requires dwave-system + token)
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from ..qubo import QUBOProblem
from .simulated_annealing import SAResult, SimulatedAnnealingSolver

logger = logging.getLogger(__name__)

# Try to import D-Wave Ocean SDK components
try:
    import dimod  # type: ignore
    _DIMOD_AVAILABLE = True
except ImportError:
    _DIMOD_AVAILABLE = False
    logger.debug("dimod not installed – D-Wave dimod sampler unavailable")

try:
    from dwave.system import DWaveSampler, EmbeddingComposite  # type: ignore
    _DWAVE_SYSTEM_AVAILABLE = True
except ImportError:
    _DWAVE_SYSTEM_AVAILABLE = False
    logger.debug("dwave-system not installed – QPU sampler unavailable")


class DWaveStubSolver:
    """
    D-Wave-compatible QUBO solver with graceful fallback.

    When Ocean SDK is present:
      - use_qpu=False  →  dimod.SimulatedAnnealingSampler (fast, local)
      - use_qpu=True   →  EmbeddingComposite(DWaveSampler()) (requires credentials)

    When Ocean SDK is absent:
      - Falls back to our own SimulatedAnnealingSolver and logs a warning.
    """

    def __init__(
        self,
        use_qpu: bool = False,
        num_reads: int = 100,
        annealing_time: int = 20,  # µs, for QPU
        seed: Optional[int] = None,
        **sampler_kwargs,
    ):
        self.use_qpu = use_qpu
        self.num_reads = num_reads
        self.annealing_time = annealing_time
        self.seed = seed
        self.sampler_kwargs = sampler_kwargs
        self._mode = self._detect_mode()

    def _detect_mode(self) -> str:
        if self.use_qpu:
            if _DWAVE_SYSTEM_AVAILABLE:
                return "qpu"
            logger.warning("use_qpu=True but dwave-system not installed; using local SA")
        if _DIMOD_AVAILABLE:
            return "dimod_sa"
        logger.warning(
            "dimod not installed – install d-wave-ocean-sdk for Ocean SDK support. "
            "Falling back to built-in simulated annealing."
        )
        return "builtin_sa"

    def solve(self, qubo: QUBOProblem) -> SAResult:
        """Solve the QUBO problem."""
        if self._mode == "builtin_sa":
            return self._solve_builtin(qubo)
        elif self._mode == "dimod_sa":
            return self._solve_dimod(qubo)
        elif self._mode == "qpu":
            return self._solve_qpu(qubo)
        else:
            return self._solve_builtin(qubo)

    def _solve_builtin(self, qubo: QUBOProblem) -> SAResult:
        solver = SimulatedAnnealingSolver(
            num_sweeps=500, num_reads=self.num_reads, seed=self.seed
        )
        return solver.solve(qubo)

    def _solve_dimod(self, qubo: QUBOProblem) -> SAResult:
        import time
        import dimod

        N = qubo.num_variables
        Q_dict = {}
        for i in range(N):
            for j in range(i, N):
                if qubo.Q[i, j] != 0:
                    Q_dict[(i, j)] = float(qubo.Q[i, j])

        bqm = dimod.BinaryQuadraticModel.from_qubo(Q_dict)
        sampler = dimod.SimulatedAnnealingSampler()
        t0 = time.perf_counter()
        response = sampler.sample(
            bqm,
            num_reads=self.num_reads,
            seed=self.seed,
            **self.sampler_kwargs,
        )
        elapsed = time.perf_counter() - t0

        best = response.first
        bits = np.array([best.sample[k] for k in range(N)], dtype=np.int8)
        return SAResult(
            bits=bits,
            energy=qubo.energy(bits),
            num_sweeps=self.num_reads,
            solve_time=elapsed,
        )

    def _solve_qpu(self, qubo: QUBOProblem) -> SAResult:
        import time
        from dwave.system import DWaveSampler, EmbeddingComposite
        import dimod

        N = qubo.num_variables
        Q_dict = {}
        for i in range(N):
            for j in range(i, N):
                if qubo.Q[i, j] != 0:
                    Q_dict[(i, j)] = float(qubo.Q[i, j])

        bqm = dimod.BinaryQuadraticModel.from_qubo(Q_dict)
        sampler = EmbeddingComposite(DWaveSampler())
        t0 = time.perf_counter()
        response = sampler.sample(
            bqm,
            num_reads=self.num_reads,
            annealing_time=self.annealing_time,
            **self.sampler_kwargs,
        )
        elapsed = time.perf_counter() - t0

        best = response.first
        bits = np.array([best.sample[k] for k in range(N)], dtype=np.int8)
        logger.info(f"QPU solve: energy={best.energy:.4f} in {elapsed:.3f}s")
        return SAResult(
            bits=bits,
            energy=qubo.energy(bits),
            num_sweeps=self.num_reads,
            solve_time=elapsed,
        )

    @property
    def backend_name(self) -> str:
        return {
            "builtin_sa": "SimulatedAnnealing (built-in)",
            "dimod_sa": "dimod.SimulatedAnnealingSampler",
            "qpu": "D-Wave QPU",
        }.get(self._mode, "unknown")

    def is_available(self) -> bool:
        return True  # always available (falls back to built-in SA)
