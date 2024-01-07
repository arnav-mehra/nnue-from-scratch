import numpy as np

class Activation:
    def __init__(self, fn_name):
        if fn_name == 'relu':
            self.function   = lambda x : np.maximum(0, x)
            self.derivative = lambda x : x >= 0

        elif fn_name == 'relu_clamped':
            self.function   = lambda x : np.clip(x, 0, 1)
            self.derivative = lambda x : 0 <= x and x <= 1

        elif fn_name == 'identity':
            self.function   = lambda x : x
            self.derivative = lambda x : 1

        elif fn_name == 'sigmoid':
            self.function   = sigmoid
            self.derivative = sigmoid_derivative

        elif fn_name == 'chessmoid':
            self.function   = chessmoid
            self.derivative = chessmoid_derivative

        elif fn_name == 'exp_normalize':
            # v_i = e^v_i / (e^v_i + c)
            # dv_i = 
            self.function = lambda v : np.exp(v)
            self.derivative = lambda v : np.exp(v)
            return

        else:
            print("ERR: Unrecognized activation function")
            return

        self.function   = np.vectorize(self.function)
        self.derivative = np.vectorize(self.derivative)