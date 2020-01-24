import random
import matplotlib.pyplot as plt

data = [name.rstrip('\n') for name in open('badges.data.txt', 'r').readlines()]
labels = [s[0] for s in data]
for i in range(len(labels)):
    if labels[i] == '+':
        labels[i] = 1
    else:
        labels[i] = 0
names = [s[2::] for s in data]

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
# Returns features in array form
def getFeatures(name):
    vowels = 'AEIOUaeiou'
    letter_count = len(name)

    letters_parity = 0
    if letter_count % 2 == 0:
        letters_parity = 1

    vowel_count = 0
    for letter in name:
        if letter in vowels:
            vowel_count += 1

    is_second_letter_vowel = 0
    if name[1] != '.':
        if name[1] in vowels:
            is_second_letter_vowel = 1
    elif name[3] in vowels:
        is_second_letter_vowel = 1

    return [letter_count, letters_parity, vowel_count, is_second_letter_vowel]

input_data = [getFeatures(name) for name in names]
X_train, Y_train = input_data[0:200], labels[0:200]
X_test, Y_test = input_data[200::], labels[200::]

# Returns an accurate weight matrix based on training data and target outputs
# Since it's impractical to wait for complete accuracy, the argument takes a iteration limit argument
def train(X, Y, iter_limit):

    iterations = 0
    learning_rate = 0.1
    weights = [round(random.uniform(-1, 1), 3) for n in X[0]]

    m = len(X)
    inaccurate = True
    while inaccurate:
        iterations += 1
        corrects = 0
        for i in range(m):
            prediction = sgnActivation(weights, X[i])
            y = Y[i]
            if prediction == y:
                corrects += 1
            else:
                diff = y - prediction
                for n in range(len(X[0])):
                    weights[n] -= learning_rate * diff * X[i][n]

        if corrects == m or iterations == iter_limit:
            inaccurate = False
            return weights

        learning_rate = learning_rate / (1 + iterations)

# Returns '+' or '-' depending on what weights the train() function learned. Will be used in score()
def predict(x, w):
    prediction_key = ['-', '+']
    prediction = sgnActivation(x, w)
    return prediction_key[prediction]

# Returns an accuracy percentage as float
def score(X, Y, W):
    checks = 0
    corrects = 0
    for i in range(len(X)):
        checks += 1
        prediction = sgnActivation(X[i], W)
        if prediction == Y[i]:
            corrects += 1
    return(corrects/checks * 100)

# Trying to plot accuracy against #iterations to see optimal parameters.
iterations = [n for n in range(2, 100)]
plt.plot(iterations, [score(X_test, Y_test, train(X_train, Y_train, i)) for i in iterations])
plt.show()
