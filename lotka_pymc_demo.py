import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
from scipy.integrate import odeint
from plot_bayesian_ode import *

print(f"Running on PyMC v{pm.__version__}")
# az.style.use("arviz-darkgrid")

# The following class encapsulates the Lotka-Volterra model
# including the ODE system, initial conditions and parameters.

class LotkaVolterra:
    def __init__(self):
        pass

    def dX_dt(self, X, t, a, b, c, d):
        return np.array([a * X[0] - b * X[0] * X[1],
                         -c * X[1] + d * b * X[0] * X[1]])

    def set_initial_conditions(self, X0):
        self.X0 = X0

    def set_parameters(self, a, b, c, d):
        self.a = a
        self.b = b
        self.c = c
        self.d = d

    def simulate(self, t, params=None):
        if params is None:
            params = (self.a, self.b, self.c, self.d)

        return odeint(self.dX_dt, y0=self.X0, t=t, args=params, rtol=0.01)




# Parámetros del modelo Lotka-Volterra
a_true = 1.0
b_true = 0.1
c = 1.5
d = 0.75
X0 = [10.0, 5.0]
size = 100
time = 15
t = np.linspace(0, time, size)

lv = LotkaVolterra()
lv.set_initial_conditions(X0)
lv.set_parameters(a_true, b_true, c, d)
# Example of simulation
# data = lv.simulate(t)
# plt.plot(t, data)


# Simulador que retorna un vector 1D
def competition_model(rng, a, b, size=None):
    lv.set_parameters()
    result = odeint(dX_dt, y0=X0, t=t, rtol=0.01, args=(float(a), float(b), c, d))
    #result = odeint(dX_dt, y0=X0, t=t, rtol=0.01, args=(float(a), float(b), c, d))
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
    # sim = pm.Simulator("sim", competition_model, params=(a, b), epsilon=10, observed=observed)
    sim = pm.Simulator("sim", lv.dX_dt, params=(a, b), epsilon=10, observed=observed)
    idata_lv = pm.sample_smc() 
    posterior = idata_lv.posterior.stack(samples=("draw", "chain"))
    post = posterior.to_pandas()

plot_observed_data(t, observed_matrix)
plot_simulations_ode(t, dX_dt, X0, observed_matrix, posterior)
