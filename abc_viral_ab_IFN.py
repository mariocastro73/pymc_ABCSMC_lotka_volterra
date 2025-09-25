
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp


def H(x: float) -> float:
    """Heaviside step function: 0 for x < 0, 1 for x >= 0."""
    return 1.0 if x >= 0.0 else 0.0


def rhs(t, y, params):
    """
    Right-hand side of the ODE system.

    Variables
    ---------
    y[0] = V(t): virus (TCID50 ml^-1)
    y[1] = A(t): neutralising antibody titre (units)
    y[2] = F(t): IFN-γ proxy (units)
    """
    V, A, F = y

    r = params["r"]          # intrinsic viral growth (d^-1)
    K = params["K"]          # carrying capacity
    phi_A = params["phi_A"]  # Ab-mediated clearance (units^-1 d^-1)
    phi_F = params["phi_F"]  # IFN-mediated clearance (units^-1 d^-1)
    k_A = params["k_A"]      # Ab expansion rate (d^-1)
    A_max = params["A_max"]  # Ab plateau
    tA_thr = params["tA_thr"]  # time Ab crosses 2-unit limit (d)
    p_F = params["p_F"]      # IFN production per virus
    d_F = params["d_F"]      # IFN decay (d^-1)

    # ODEs
    dVdt = r * V * (1.0 - V / K) - phi_A * A * V - phi_F * F * V
    dAdt = k_A * (A_max - A) * H(t - tA_thr)
    dFdt = p_F * V - d_F * F

    return [dVdt, dAdt, dFdt]


def integrate_system(
    t_span,
    y0,
    params,
    t_eval=None,
    rtol=1e-8,
    atol=1e-10,
    method="RK45",
):
    """
    Integrate the system using scipy's solve_ivp.

    Parameters
    ----------
    t_span : tuple(float, float)
        (t0, tfinal)
    y0 : array-like
        Initial conditions [V0, A0, F0]
    params : dict
        Model parameters (see rhs)
    t_eval : array-like or None
        Times at which to store the computed solution.
    rtol, atol : float
        Solver tolerances.
    method : str
        Integration method for solve_ivp.

    Returns
    -------
    sol : OdeResult
        SciPy solution object with sol.t, sol.y
    """
    if t_eval is None:
        t_eval = np.linspace(t_span[0], t_span[1], 400)

    sol = solve_ivp(
        fun=lambda t, y: rhs(t, y, params),
        t_span=t_span,
        y0=y0,
        t_eval=t_eval,
        method=method,
        rtol=rtol,
        atol=atol,
        vectorized=False,
        dense_output=False,
    )
    if not sol.success:
        raise RuntimeError(f"ODE integration failed: {sol.message}")
    return sol


def plot_solution(sol):
    """
    Plot V, A, F on a semilog-y axis.
    Handles zeros by adding a tiny epsilon for visualization only.
    """
    t = sol.t
    V, A, F = sol.y
    eps = 1e-8  # to avoid log(0) in visualization

    plt.figure(figsize=(8, 5))
    plt.semilogy(t, V + eps, label="Virus, V(t)")
    plt.semilogy(t, A + eps, label="Antibody, A(t)")
    plt.semilogy(t, F + eps, label="IFN-γ proxy, F(t)")
    plt.ylim(1e-1, 1e7)

    plt.xlabel("Time (days)")
    plt.ylabel("Level (log scale)")
    plt.title("Within-host dynamics with innate (F) and adaptive (A) immunity")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Example parameters (choose sensible placeholders; tune as needed)
    params = {
        "r": 12,        # d^-1
        "K": 1e6,        # carrying capacity
        "phi_A": 1e-3,   # units^-1 d^-1
        "phi_F": 0,   # units^-1 d^-1
        "k_A": 5.,      # d^-1
        "A_max": 1e3,    # units
        "tA_thr": 5.0,   # days
        "p_F": 1e-3,     # IFN per virus
        "d_F": 1.0,      # d^-1
    }

    # Initial conditions at start of integration (e.g., when V > 2 log10)
    V0 = 1e2     # TCID50 ml^-1
    A0 = 0     # just below detection threshold (units)
    F0 = 0.0     # baseline IFN proxy
    y0 = [V0, A0, F0]

    # Time span (days)
    t0, tf = 0.0, 20.0
    t_eval = np.linspace(t0, tf, 600)

    # Integrate
    sol = integrate_system(t_span=(t0, tf), y0=y0, params=params, t_eval=t_eval)

    # Plot
    plot_solution(sol)


