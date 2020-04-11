class Perceptron(object):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def train(self, epoch_limit, learning_rate, trace_table = False):
        def dotProduct(m1, m2):
            product = sum([(m1[i] * m2[i]) for i in range(len(m1))])
            return product

        def sgnActivation(w, x):
            wx = dotProduct(w, x)
            if wx > 0:
                return 1
            else:
                return 0

        bias = 1
        weights = [0 for n in range(len(self.X[0]))]
        m = len(self.X)
        inaccurate = True
        epoch = 0
        if trace_table:
            print('Epoch, Input Vector, Weight Vector, Target, Prediction')

        while inaccurate:
            epoch += 1
            corrects = 0
            for i in range(m):
                prediction = sgnActivation(weights, self.X[i])
                y = self.Y[i]
                if prediction == y:
                    corrects += 1
                else:
                    diff = y - prediction
                    for n in range(len(self.X[0])):
                        weights[n] += learning_rate * diff * self.X[i][n]
                    # bias += learning_rate * diff

                if trace_table:
                    print(f"{epoch}, {self.X[i]}, {weights}, {y}, {prediction}")

            if corrects == m or epoch == epoch_limit:
                inaccurate = False
                return weights

    def predict(self, X):
        pass

    def score():
        pass