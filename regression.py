import numpy as np
import pandas as pd

flight_data = pd.read_csv('UA.csv')

flight_data.dropna(axis=1, how='all', inplace=True)
flight_data.dropna(axis=0, how='any', subset=[
    'AIR_SYSTEM_DELAY',	'SECURITY_DELAY',	'AIRLINE_DELAY',	'LATE_AIRCRAFT_DELAY'	, 'WEATHER_DELAY'], inplace=True)

feature = flight_data['DISTANCE']
target = flight_data['ELAPSED_TIME']

weight = np.random.randn()
bias = np.random.randn()
print(weight == bias)

training_num = 80 * len(feature) / 100
train_x = feature[:training_num]
train_y = target[:training_num]

test_x = feature[training_num + 1:]
test_y = target[training_num + 1:]
