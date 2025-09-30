import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
from scipy.integrate import odeint
from plot_bayesian_ode import *

print(f"Running on PyMC v{pm.__version__}")
# az.style.use("arviz-darkgrid")

# Parámetros del modelo Lotka-Volterra
a_true = 1.0
b_true = 0.1
c = 1.5
d = 0.75
X0 = [10.0, 5.0]
size = 100
time = 15
t = np.linspace(0, time, size)

# Ecuaciones del sistema
def dX_dt(X, t, a, b, c, d):
    return np.array([a * X[0] - b * X[0] * X[1],
                     -c * X[1] + d * b * X[0] * X[1]])

# Simulador que retorna un vector 1D
def competition_model(rng, a, b, size=None):
    result = odeint(dX_dt, y0=X0, t=t, rtol=0.01, args=(float(a), float(b), c, d))
    return result.reshape(-1)

# Generación de datos observados con ruido
def add_noise(a, b):
    noise = np.random.normal(scale=2, size=(size, 2))
    simulated = odeint(dX_dt, y0=X0, t=t, rtol=0.01, args=(a, b, c, d))
    return (simulated + noise).reshape(-1)

# Datos observados
observed = add_noise(a_true, b_true)
observed_matrix = observed.reshape(size, 2)

# Modelo probabilístico con PyMC
with pm.Model() as model_lv:
    a = pm.HalfNormal("a", 1.0)
    b = pm.HalfNormal("b", 1.0)
    sim = pm.Simulator("sim", competition_model, params=(a, b), epsilon=10, observed=observed)
    idata_lv = pm.sample_smc() 
    posterior = idata_lv.posterior.stack(samples=("draw", "chain"))
    post = posterior.to_pandas()

plot_observed_data(t, observed_matrix)
plot_simulations_ode(t, dX_dt, X0, observed_matrix, posterior)
