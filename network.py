"""
network.py

This is the core dynamical system used throughout this project, aka a random
recurrent neural network with a single tunable gain parameter.

    x(t+1) = tanh( g * W @ x(t)  +  w_in * u(t) )

    W @ x(t)      -- for every neuron a weighted sum of every other
                     neuron's current activity ("W" = recurrent weights).
    g * (...)     -- that weighted sum is then scaled by a single number
                     g, the "gain". This is the ONE parameter we will
                     vary throughout the whole project.
    + w_in * u(t) -- an external scalar input u(t) injected into the
                     network
    tanh( ... )   -- tanh nonlinearity, keeps every neuron's activity
                     between -1 and +1 no matter how large the input to
                     it was.

G purpose: g scales how strongly neurons influence each
other through W.:

    * g very small  -> each neuron barely listens to the others -> the
                        network's activity quickly dies down to (near)
                        zero and forgets everything almost instantly.
                        This is the ORDERED regime.
    * g very large   -> each neuron is very strongly influenced by many
                        others -> tiny differences get amplified over and
                        over -> the network's trajectory becomes wildly
                        sensitive to minuscule details of its state and
                        input history. This is the CHAOTIC regime.
    * g just right    -> somewhere in between, there is a specific value
                        of g where the network transitions
                        from one behaviour to the other. This transition
                        point is the EDGE OF CHAOS that the whole
                        project is about.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class RandomRNN:
    """
    single random recurrent neural network ("reservoir") with a
    tunable scalar gain.

    Parameters
    ----------
    n_units : int
    gain : float
    seed : int, default 0
    input_scale : float, default 1.0

    Attributes
    ----------
    W : np.ndarray of shape (n_units, n_units)
    w_in : np.ndarray of shape (n_units,)
    state : np.ndarray of shape (n_units,)
    """

    n_units: int
    gain: float
    seed: int = 0
    input_scale: float = 1.0

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)

        # --- Recurrent weight matrix W 
        self.W: np.ndarray = rng.normal(
            loc=0.0,
            scale=1.0 / np.sqrt(self.n_units),
            size=(self.n_units, self.n_units),
        )

        # --- Input weight vector w_in 
        self.w_in: np.ndarray = self.input_scale * rng.normal(
            loc=0.0, scale=1.0, size=self.n_units
        )

        # --- Initial state
        self.state: np.ndarray = np.zeros(self.n_units)

    # ------------------------------------------------------------------
    # Simulation methods
    # ------------------------------------------------------------------

    def reset_state(self, x0: np.ndarray | None = None) -> None:
        """
        Reset of the network's state.
        """
        if x0 is None:
            self.state = np.zeros(self.n_units)
        else:
            x0 = np.asarray(x0, dtype=float)
            if x0.shape != (self.n_units,):
                raise ValueError(
                    f"x0 must have shape ({self.n_units},), got {x0.shape}"
                )
            self.state = x0.copy()

    def step(self, u_t: float = 0.0) -> np.ndarray:
        """
        Advances the network by exactly one discrete time step.

        Implements:
            x(t+1) = tanh( gain * (W @ x(t)) + w_in * u_t )
        and updates ``self.state`` in place.

        Parameters
        ----------
        u_t : float

        Returns
        -------
        np.ndarray of shape (n_units,)
            The new state x(t+1).
        """
        pre_activation = self.gain * (self.W @ self.state) + self.w_in * u_t
        self.state = np.tanh(pre_activation)
        return self.state

    def run(
        self, u: np.ndarray, x0: np.ndarray | None = None
    ) -> np.ndarray:
        """
        Simulates the network forward through an entire input sequence.

        Parameters
        ----------
        u : np.ndarray of shape (T,)
            Sequence of scalar inputs: u[0], u[1], ..., u[T-1].
        x0 : np.ndarray of shape (n_units,), optional
            Initial state. Defaults to all zeros (see reset_state).

        Returns
        -------
        states : np.ndarray of shape (T, n_units)
            The full trajectory of states.
        """
        self.reset_state(x0)
        u = np.asarray(u, dtype=float)
        T = len(u)
        states = np.empty((T, self.n_units))
        for t in range(T):
            states[t] = self.step(u[t])
        return states

    # ------------------------------------------------------------------
    #linearisation around the origin
    # ------------------------------------------------------------------

    def jacobian_at_origin(self) -> np.ndarray:
        """
        Returns the Jacobian matrix of the input-free update rule,
        linearised around the fixed point x = 0

        Near x = 0, F(x) = tanh(g * W @ x) can be approximated by its
        first-order (linear) Taylor expansion. Since d/dx[tanh(x)] at
        x=0 equals 1, this linear approximation:

            F(x) ~= J @ x,     with     J = g * W

        the Jacobian of the system at the origin is just g times the
        (fixed, random) recurrent weight matrix W. As g increases, every
        eigenvalue of J = g*W is scaled up by exactly the same factor g
        - so the shape of the eigenvalue spectrum never changes, only
        its overall size. 
        """
        return self.gain * self.W

    # debugging

    def __repr__(self) -> str:  # pragma: no cover - cosmetic only
        return (
            f"RandomRNN(n_units={self.n_units}, gain={self.gain}, "
            f"seed={self.seed}, input_scale={self.input_scale})"
        )


