"""
Implementation of the Perceptron algorithm. Learns the  boolean OR function
Has a dynamic learning rate. Decreases every epoch (iteration)
"""

import random

# We will use this to multiply input and weight vectors
def dotProduct(m1, m2):
    product = sum([(m1[i] * m2[i]) for i in range(len(m1))])
    return product

# Returns 1/0 based on dot product
def sgnActivation(w, x):
    wx = dotProduct(w, x)
    if wx > 0:
        return 1
    else:
        return 0

"""
                          --- Initialisation ---
"""

# Input array with format [input1, input2] for OR function
X = [
[0, 0],
[0, 1],
[1, 0],
[1, 1],
]

# Target outputs
targets = [0, 1, 1, 1]

# Initialising each weight such that -1 < w < 1
# W_ij where i = input value; j = input dimension
weights = [round(random.uniform(-1, 1)), round(random.uniform(-1, 1))]
# weights = [[round(random.uniform(-1, 1), 3),
#             round(random.uniform(-1, 1), 3)] for item in X]

"""
                          --- Training ---
"""

iterations = 0
learning_rate = 0.25  # This is the starting learning rate. We will be updrating it per epoch
not_accurate = True
m = len(X)
while not_accurate:
    iterations += 1
    corrects = 0
    # Looping through each pair of weights and inputs
    for i in range(m):
        y = sgnActivation(X[i], weights)
        t = targets[i]
        if y == t:
            corrects += 1
        else:
            diff = y - t
            for n in range(2):
                # Updating weights based on difference between target and actual output
                weights[n] -= learning_rate * diff * X[i][n]
            print('\n',iterations)
            print('Current Weights: ', weights)
            print('Current Activations: ', [sgnActivation(weights, X[i]) for i in range(m)])

    if corrects == 4:
        # Recall
        print('\n• Final Weights: ', weights)
        print('• Iterations: ', iterations)
        print('• Final Activations: ',[sgnActivation(weights, X[i]) for i in range(m)])
        not_accurate = False
    else:
        continue

    # Decreasing the learning rate with the function n1 = n0 / (1 + epochs)
    learning_rate = learning_rate / (1 + iterations)
