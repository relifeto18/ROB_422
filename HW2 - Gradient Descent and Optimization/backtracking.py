def backtracking(func, grad_func, x, dx, t = 1, alpha = 0.1, beta = 0.6):
    while (func(x + t * dx) > func(x) + alpha * t * grad_func(x) * dx):    
        t *= beta
    return t
