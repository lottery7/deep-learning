import numpy as np
from itertools import product


def nn(data, weights):
    output = np.dot(data, weights)
    return output


data_to_learn = np.array([
    [1, 0, 1],
    [0, 1, 1],
    [0, 0, 1],
], dtype=np.float32)

outputs_to_learn = np.array([0, 1, 0], dtype=np.float32)

weights = np.random.rand(3, 1)

alpha = 0.1

for j in range(30):
    for i in range(len(data_to_learn)):
        data = data_to_learn[i].reshape(1, 3)
        expected_output = outputs_to_learn[i].reshape(1, 1)
        output = nn(data=data, weights=weights)

        error = (expected_output - output) ** 2
        weights_deltas = data.T * (expected_output - output) * alpha

        weights += weights_deltas


for data in product([0, 1], repeat=3):
    data = np.array(data).reshape(1, 3)
    output = nn(data=data, weights=weights)

    print(f"Output: {output[0, 0]} Expected: {data[0, 1]}")
