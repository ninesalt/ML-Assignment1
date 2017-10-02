from numpy import random, dot, array, arange
from pandas import read_csv
import matplotlib.pyplot as plt

credit_cards = read_csv('creditcards.csv')

features = array(credit_cards[['V2', 'V11']])
targets = array(credit_cards['Class'])

weights = random.randn(2)
bias = random.randn()

learning_rate = 0.2
error_tolerance = 0.14     # 0.14% error tolerance
iterations = 0
mispredictions = 100  # 100% wrong predictions

while mispredictions > error_tolerance:

    iterations += 1
    errors = 0

    for i in range(len(features)):

        inputs = features[i]
        expected = targets[i]

        summation = dot(inputs, weights) - bias
        result = 1 if summation > 0 else 0

        error = result - expected

        weights -= learning_rate * error * inputs
        bias -= learning_rate * error * -1

        if error != 0:
            errors += 1

    # percent of wrong guesses
    mispredictions = errors * 100 / len(features)

    print('percent of mispredictions in iteration {} : {}% '.format(
        iterations, mispredictions))

print('Converged with learning rate of {} after {} iterations with an error tolerance of {}%'.format(
    learning_rate, iterations, error_tolerance))


# plotting
v2 = [value for value, _ in features]
v11 = [value for _, value in features]

w0 = bias
w1, w2 = weights
x = arange(0, 20)
y = (w0 / w2) - (w1 * x / w2)

colors = [value / 255 for value in targets]

plt.scatter(v2, v11, c=colors)
plt.plot(x, y, 'k-')
plt.show()
