"""
Simulated Annealing solver for QUBO problems.

This is the primary classical backend.  Simulated annealing is a well-known
metaheuristic that maps naturally onto QUBO: flipping a single binary variable
changes the energy by ΔE = (1-2*x_k) * (Q[k,k] + 2*Σ_{l≠k} Q[k,l]*x_l),
which can be computed in O(N) time per flip with a cached effective-field vector.

Algorithm:
  1. Start from a random binary vector
  2. At each step, pick a random variable and compute ΔE of flipping it
  3. Accept the flip if ΔE < 0 (improvement) or with probability exp(-ΔE/T)
  4. Anneal temperature T from T_start → T_end over num_sweeps sweeps
  5. Return the best solution seen during the run
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..qubo import QUBOProblem


@dataclass
class SAResult:
    """Result from a simulated annealing run."""
    bits: np.ndarray          # Best binary solution found
    energy: float             # QUBO energy of best solution
    num_sweeps: int           # Sweeps completed
    solve_time: float         # Wall-clock seconds
    energy_history: list = field(default_factory=list)  # energy per sweep


class SimulatedAnnealingSolver:
    """
    Simulated Annealing solver for QUBO problems.

    Parameters
    ----------
    num_sweeps:
        Number of full sweeps (N variable updates) to perform.
    num_reads:
        Number of independent SA runs; best result is kept.
    T_start:
        Initial temperature. A good default is the max |Q_ij| value.
    T_end:
        Final temperature (close to 0).
    schedule:
        Annealing schedule: 'geometric' or 'linear'.
    seed:
        Random seed for reproducibility.
    """

    def __init__(
        self,
        num_sweeps: int = 1000,
        num_reads: int = 10,
        T_start: Optional[float] = None,
        T_end: float = 0.01,
        schedule: str = "geometric",
        seed: Optional[int] = None,
    ):
        self.num_sweeps = num_sweeps
        self.num_reads = num_reads
        self.T_start = T_start
        self.T_end = T_end
        self.schedule = schedule
        self.rng = np.random.default_rng(seed)

    def solve(self, qubo: QUBOProblem) -> SAResult:
        """Run SA on the given QUBO problem."""
        N = qubo.num_variables
        Q = qubo.Q
        # Build a symmetric effective matrix for energy computation
        # Energy = x^T Q x  where Q is upper-triangular
        # For field computation we need the full symmetric form
        Q_sym = Q + Q.T - np.diag(np.diag(Q))   # symmetrise

        T_start = self.T_start
        if T_start is None:
            T_start = float(np.max(np.abs(Q_sym))) * 2.0
            if T_start < 1e-9:
                T_start = 1.0

        best_bits = None
        best_energy = float("inf")
        energy_history: list = []

        t0 = time.perf_counter()

        for _ in range(self.num_reads):
            bits = self.rng.integers(0, 2, size=N, dtype=np.int8)
            # Effective field: h_k = Σ_l Q_sym[k,l] * x_l
            h = Q_sym @ bits.astype(float)

            current_energy = float(bits @ h)  # = x^T Q_sym x (= 2 * QUBO energy for off-diag)
            # Actually use proper QUBO energy
            current_energy = qubo.energy(bits)

            sweep_energies = []

            for sweep in range(self.num_sweeps):
                # Temperature schedule
                if self.schedule == "geometric":
                    frac = sweep / max(self.num_sweeps - 1, 1)
                    T = T_start * (self.T_end / T_start) ** frac
                else:  # linear
                    T = T_start - (T_start - self.T_end) * sweep / max(self.num_sweeps - 1, 1)
                T = max(T, 1e-12)

                # One full sweep: try flipping each variable in random order
                order = self.rng.permutation(N)
                for k in order:
                    # ΔE = energy_flip - energy_current
                    # For QUBO: ΔE = (1-2x_k) * (Q[k,k] + Σ_{l≠k} (Q[k,l]+Q[l,k])*x_l)
                    flip_sign = 1 - 2 * int(bits[k])
                    delta_e = flip_sign * (Q_sym[k] @ bits.astype(float))

                    if delta_e < 0 or self.rng.random() < np.exp(-delta_e / T):
                        bits[k] ^= 1
                        current_energy += delta_e

                sweep_energies.append(current_energy)

            # Recompute exact energy to avoid floating-point drift
            exact_energy = qubo.energy(bits)
            if exact_energy < best_energy:
                best_energy = exact_energy
                best_bits = bits.copy()

        energy_history = sweep_energies  # from last read; good enough for plotting

        return SAResult(
            bits=best_bits,
            energy=best_energy,
            num_sweeps=self.num_sweeps * self.num_reads,
            solve_time=time.perf_counter() - t0,
            energy_history=energy_history,
        )
