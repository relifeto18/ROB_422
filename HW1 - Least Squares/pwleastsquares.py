import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=4)

position = np.loadtxt('calibration.txt', delimiter=' ')
command = position[:, 0]   # get commanded position
measure = position[:, 1]   # get measured position

idx1 = np.where(command == -0.5)[0][0]   # get index for knotpoint = -0.5
idx2 = np.where(command == 0.5)[0][0]   # get index for knotpoint = 0.5

# add constraints to force the lines intersect at the knotpoints
commanded = command.reshape(-1, 1)   # add row for knotpoint intersection 
A = np.pad(np.hstack((commanded, np.ones(commanded.shape))), [(0, 0), (0, 4)], constant_values = 0)   # expand the matrix to handle multiple curves 
A[idx1 : idx2 + 1, 2 : 4] = A[idx1 : idx2 + 1, 0 : 2]   # move to third and forth columns for positions inside the range [-0.5, 0.5]
A[idx2 + 1 : , 4 : ] = A[idx2 + 1 : , 0 : 2]   # move to fifth and sixth columns for positions larger than 0.5
A[idx1 : , 0 : 2] = 0   # first two columns only for positions smaller than -0.5
C = np.array([[-0.5, 1, 0.5, -1 ,0, 0], [0, 0, -0.5, -1, 0.5, 1]])   # create constraints 
Z = np.array([0, 0])
measured = np.append(A.T @ measure, Z, axis=0)   # construct [A^Tb Z]T 8x1 matrix
A_new = np.append(np.append(A.T @ A, C.T, axis=1), np.append(C, np.zeros((2, 2)), axis=1), axis=0) # construct [A^TA C^T; C 0] 8x8 matrix

x = (np.linalg.inv((A_new.T @ A_new)) @ A_new.T @ measured)[:-2]   # least square parameters [x1, x2, x3, x4, x5, x6, lambda1, lambda2]
predict = A @ x   # predict position
SSE = np.sum((predict - measure)**2)   # sum of squared error

# predict position for command at 0.68
predicted = x[4] * 0.68 + x[5]

print("The parameter values:", np.array2string(x.reshape(-1, 1), prefix="The parameter values: "))
print("The sum of squared errors: ", np.array2string(SSE))
print("The prediction for 0.68.: ", np.array2string(predicted))

# plot 
plt.scatter(command, measure, color="blue", label = "Commanded vs measured")
plt.plot(command, predict, color="red", label = "Fitted line")
plt.legend()
plt.show()