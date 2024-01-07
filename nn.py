import numpy as np
import swifter
from activation import Activation
from layer import Layer

class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers

    def predict(self, x):
        for layer in self.layers:
            x = layer.predict(x) # pipe result to next layer
        return x

    def predict_batch(self, inputs):
        return inputs.swifter.progress_bar(False).apply(self.predict)

    def cost(self, x, s):
        z = self.predict(x)
        delta = z - s
        delta = delta.T[0] # column vector -> array
        return np.dot(delta, delta)

    def batch_average_cost(self, inputs, solutions):
        output = self.predict_batch(inputs)
        sq_errors = np.power(output - solutions, 2.0)
        avg_sq_errs = np.average(sq_errors)
        return np.sum(avg_sq_errs)

    def backprop(self, x, s):
        z = self.predict(x) # gets result and sets prev_x/y/z

        # C = self_dot(z - s)
        delta = z - s
        C = np.dot(delta.T[0], delta.T[0])

        # dC/dz_i = d/dz_i (z_i - s_i)^2 = 2(z_i - s_i)
        dC_dz = 2.0 * delta

        adjustments = []
        for layer in reversed(self.layers):
            dC_db, dC_dA, dC_dx = layer.backprop(dC_dz)
            adjustments.append((dC_db, dC_dA))
            dC_dz = dC_dx # pipe to next layer
        adjustments.reverse()
        return adjustments

    def backprop_batch(self, df, in_field, out_field):
        return df.swifter.progress_bar(False).apply(
            lambda r: self.backprop(r[in_col], r[out_col]),
            axis=1
        )

    def process_batch(self, df, in_field, out_field):
        n = len(df)

        in_col = df.columns.get_loc(in_field)
        out_col = df.columns.get_loc(out_field)
        
        results = self.backprop_batch(df, in_field, out_field)
        results = np.array(results)

        # results[sample][layer][param]
        # adjs[layer][param]

        def avg_layer_params(l):
            dC_db = np.copy(results[0][l][0])
            dC_dA = np.copy(results[0][l][1])
            for s in range(1, n):
                dC_db += results[s][l][0]
                dC_dA += results[s][l][1]
            dC_db /= n
            dC_dA /= n
            return ( dC_db, dC_dA )

        adjs = [avg_layer_params(l) for l in range(len(self.layers))]
        return adjs

    def apply_adjustments(self, adjs, alpha):
        for i, ( dC_db, dC_dA ) in enumerate(adjs):
            layer = self.layers[i]
            layer.b -= alpha * dC_db
            layer.A -= alpha * dC_dA

    def train(
        self, 
        df, in_col, out_col,
        batch_size = 512, alpha = 1.0e-2,
        stopper = lambda err, i : i == 100
    ):
        i = 0

        while True:
            sample = df.sample(n = batch_size)
            ins = sample[in_col]
            outs = sample[out_col]
            adjs = self.process_batch(sample, in_col, out_col)

            err = self.batch_average_cost(ins, outs)
            print("perror:", err)

            self.apply_adjustments(adjs, alpha)
            err = self.batch_average_cost(ins, outs)
            print("ferror:", err)

            i += 1
            if stopper(err, i):
                break

    def model_stats(self, ins, sol, center):
        pred = self.predict_batch(ins)
        
        abs_errors = np.absolute(pred - sol)[0]
        avg_abs_err = np.average(abs_errors)[0]
        avg_sq_err = np.dot(abs_errors[0], abs_errors[0])[0]
        sign_acc = 1 - np.sum((pred > center) ^ (sol > center)) / len(pred)
    
        print(f"AVERAGE ABSOLUTE ERROR: {avg_abs_err}")
        print(f"AVERAGE SQUARE ERROR:   {avg_sq_err}")
        print(f"SIGN ACCURACY:          {sign_acc}")

    def push_backup(self):
        for layer in self.layers:
            layer.push_backup()

    def pop_backup(self):
        for layer in enumerate(self.layers):
            layer.pop_backup()