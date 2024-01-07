import numpy as np
import swifter
from activation import Activation

class Layer:
    def __init__(self, inputs, outputs, activation_fn_name):
        self.activation = Activation(activation_fn_name)
        self.inputs = inputs
        self.outputs = outputs
        
        max_weight = np.sqrt(2.0 / self.inputs)
        self.A = np.random.rand(outputs, inputs) * max_weight
        self.b = np.zeros((outputs, 1))

        self.backups = []

    def predict(self, x): # forward prop
        x = np.array(x).reshape(self.inputs, 1)
        y = np.matmul(self.A, x) + self.b # y = Ax + b
        z = self.activation.function(y)   # z = f(y)

        self.prev_x = x
        self.prev_y = y
        self.prev_z = z

        return z        

    def backprop(self, dC_dz): # backwards prop
        # assuming: z_i = o(y_i)
        # dz_i/dy_i = o'(y_i)
        dz_dy = self.activation.derivative(self.prev_y)
        
        # dC/dy_i = dC/dz_i * dz_i/dy_i
        dC_dy = dC_dz * dz_dy

        # dC/db_i = dC/dy_i * dy/db_i = dC/dy_i * 1
        dC_db = dC_dy

        # dC/dA_ij = dC/dy_i * dy_i/dA_ij = dC/dy_i * x_j
        # dC_dA = np.empty((self.outputs, self.inputs))
        # for r in range(0, self.outputs):
        #     for c in range(0, self.inputs):
        #         dC_dA[r][c] = dC_dy[r][0] * self.prev_x[c][0]
        dC_dA = np.matmul(dC_dy, self.prev_x.T)

        # if not np.allclose(dC_dA, dC_dA_2):
        #     print("shit")

        # dC/dx_i = dC/dy_i * dy_i/dx_i = dC/dy_i * A_ij
        # dC_dx = np.zeros((self.inputs, 1))
        # for c in range(0, self.inputs):
        #     for r in range(0, self.outputs):
        #         dC_dx[c][0] += dC_dy[r][0] * self.A[r][c]
        dC_dx = np.matmul(self.A.T, dC_dy)

        # if not np.allclose(dC_dx, dC_dx_2):
        #     print("shit")

        return dC_db, dC_dA, dC_dx

    def push_backup(self):
        self.backups.append((
            np.copy(self.A),
            np.copy(self.b)
        ))

    def pop_backup(self, nParams):
        (nA, nb) = self.backups.pop()
        self.A = nA
        self.b = nb