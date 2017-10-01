import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

credit_cards = pd.read_csv('creditcards.csv')

features = np.array(credit_cards[['V2', 'V11']])
targets = np.array(credit_cards['Class'])

weights = np.random.randn(2)
bias = np.random.randn()
learning_rate = 0.2
errors = []

while True:
    for i in range(features.shape[0]):

        inputs = features[i]
        expected = targets[i]

        summation = np.dot(inputs, weights) + bias
        result = 1 if summation < 0 else 0

        error = result - expected
        weights -= learning_rate * error * inputs
        bias -= learning_rate * error * -1

        errors.append(error)
        print(result, expected, result == expected, np.mean(errors))
