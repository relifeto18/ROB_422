import numpy as np
import cvxpy as cp

#these are defined as [a b]
hyperplanes = np.mat([[0.7071,    0.7071, 1.5], 
                [-0.7071,    0.7071, 1.5],
                [0.7071,    -0.7071, 1],
                [-0.7071,    -0.7071, 1]])

#the optimization function c:
c = np.mat([2, 1]).T

#let's break down the data into variables we are familiar with
A = hyperplanes[:][:,0:2] # each column is the "a" part of a hyperplane
b = hyperplanes[:,2] # each row is the "b" part of a hyperplane (only one element in each row)

m, n = A.shape

x = cp.Variable((n, 1))
prob = cp.Problem(cp.Minimize(c.T@x),
                 [A @ x <= b])
prob.solve()

print("The optimal point:", np.array2string(x.value, prefix="The optimal point: "))
print("The optimal value is:", prob.value)