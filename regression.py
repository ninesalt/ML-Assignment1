import numpy as np
import pandas as pd
from math import tanh


def normalize(vector):
    return np.divide(vector, np.sqrt(sum(vector ** 2)))


flight_data = pd.read_csv('UA.csv')
flight_data.dropna(axis=0, how='any', subset=[
                   'DISTANCE', 'ELAPSED_TIME'], inplace=True)

feature = normalize(np.array(flight_data['DISTANCE']))
target = normalize(np.array(flight_data['ELAPSED_TIME']))
learning_rate = 0.2

weight = np.random.randn()
bias = np.random.randn()

# integer division to automatically floor
training_num = 80 * len(feature) // 100

train_x = feature[:training_num]
train_y = target[:training_num]

test_x = feature[training_num:]
test_y = target[training_num:]

mean_errors = []

# Training
for _ in range(10):

    for i in range(len(train_x)):

        inputs = train_x[i]
        actual = train_y[i]

        summation = np.dot(inputs, weight) - bias

        # predicted = 1 / (1 + np.exp(-summation))  # sigmoid function
        predicted = tanh(summation)
        error = predicted - actual

        weight -= learning_rate * error * inputs
        bias -= learning_rate * error * -1

    # mean_errors.append(error**2)
    # mean = np.mean(mean_errors)
    # print(mean)


# Testing

mean_errors = []

for i in range(len(test_x)):

    inputs = test_x[i]
    actual = test_y[i]

    summation = np.dot(inputs, weight) - bias
    predicted = tanh(summation)
    error = predicted - actual
    mean_errors.append(error ** 2)

print('Mean square of errors: ', np.mean(mean_errors))
