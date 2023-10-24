import numpy as np

def sgd(fiprime, x, maxi):
    t = 1
    # max_iter = 750
    max_iter = 1000
    iter = 0
    x_out = [x]
    for iter in range(max_iter):
        i = np.random.randint(1, maxi)
        dx = -fiprime(x, i)
        x += t * dx
        x_out.append(x)
    return np.array(x_out)