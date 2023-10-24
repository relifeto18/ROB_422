import backtracking
import numpy as np

def GD(func, grad_func, x, e = 0.0001):
    x_out = []
    x_out.append(x)
    dx = -grad_func(x)
    while (np.abs(dx) > e):
        t = backtracking.backtracking(func, grad_func, x, dx)
        x += t * dx
        x_out.append(x)
        dx = -grad_func(x)
    return np.array(x_out)