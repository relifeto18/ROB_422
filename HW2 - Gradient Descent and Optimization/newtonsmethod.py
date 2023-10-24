import backtracking
import numpy as np

def Newton(func, grad_func, hessian, x, e = 0.0001):
    x_out = []
    x_out.append(x)
    while True:
        dx = -1 / hessian(x) * grad_func(x)
        lam = -grad_func(x) * dx
        if (0.5 * lam <= e):
            return np.array(x_out)
        else:
            t = backtracking.backtracking(func, grad_func, x, dx)
            x += t * dx
            x_out.append(x)