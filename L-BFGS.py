import numpy as np
from scipy.optimize import fmin_l_bfgs_b

X = np.arange(0, 10, 1)
M = 2
B = 3
Y = M * X + B


def func(parameters, *args):
    x = args[0]
    y = args[1]
    m, b = parameters
    y_model = m * x + b
    error = sum(np.power((y - y_model), 2))
    return error


initial_values = np.array([0.0, 1.0])

x_opt, f_opt, info = fmin_l_bfgs_b(func, x0=initial_values, args=(X, Y),
                                   approx_grad=True)

print(x_opt, f_opt)
