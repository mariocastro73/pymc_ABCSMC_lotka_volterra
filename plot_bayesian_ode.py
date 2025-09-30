import arviz as az
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint

def plot_observed_data(t, observed_matrix):
    _, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t, observed_matrix[:, 0], "x", label="prey")
    ax.plot(t, observed_matrix[:, 1], "x", label="predator")
    ax.set_xlabel("time")
    ax.set_ylabel("population")
    ax.set_title("Observed data")
    ax.legend()
    plt.show()
    return ax

def plot_simulations_ode(t, dX_dt, X0, observed_matrix, posterior):
    _, ax = plt.subplots(figsize=(14, 6))
    ax.plot(t, observed_matrix[:, 0], "o", label="prey", c="C0", mec="k")
    ax.plot(t, observed_matrix[:, 1], "o", label="predator", c="C1", mec="k")

    mean_a = posterior["a"].mean().item()
    mean_b = posterior["b"].mean().item()
    mean_sim = odeint(dX_dt, y0=X0, t=t, rtol=0.01, args=(mean_a, mean_b, c, d))
    ax.plot(t, mean_sim[:, 0], linewidth=3, label="mean prey", c="C0")
    ax.plot(t, mean_sim[:, 1], linewidth=3, label="mean predator", c="C1")

    for i in np.random.randint(0, posterior.samples.size, 75):
        ai = posterior["a"].values[i]
        bi = posterior["b"].values[i]
        sim_i = odeint(dX_dt, y0=X0, t=t, rtol=0.01, args=(ai, bi, c, d))
        ax.plot(t, sim_i[:, 0], alpha=0.1, c="C0")
        ax.plot(t, sim_i[:, 1], alpha=0.1, c="C1")

    ax.set_xlabel("time")
    ax.set_ylabel("population")
    ax.legend()
    plt.show()
    return ax


def plot_posterior(idata_lv):
    az.plot_trace(idata_lv, kind="rank_vlines")
    plt.show()
    az.plot_posterior(idata_lv)
    plt.show()

    return ax





