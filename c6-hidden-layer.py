import numpy as np
from itertools import product

data_to_learn = np.array([
    [ 1, 0, 1 ],
    [ 0, 1, 1 ],
    [ 0, 0, 1 ],
    [ 1, 1, 1 ]
], dtype=np.float32)

outputs_to_learn = np.array([1, 1, 0, 0], dtype=np.float32)

alpha = 0.2

np.random.seed(1)
hidden_size = 4
weights = [np.random.rand(3, hidden_size) * 2 - 1, np.random.rand(hidden_size, 1) * 2 - 1]


def relu(x):
    return (x > 0) * x

def relu_deriv(x):
    return x > 0


def nn(data, *weights):
    def relu(x): return (x > 0) * x
    layers = [data] + [None] * len(weights)

    layers[1] = relu(np.dot(layers[0], weights[0]))
    layers[2] = np.dot(layers[1], weights[1])

    return layers

for i in range(60):
    for j in range(len(data_to_learn)):
        data = data_to_learn[j].reshape(1, 3)
        expect = outputs_to_learn[j].reshape(1, 1)

        layers = nn(data, *weights)

        layer_2_delta = (expect - layers[2])
        layer_1_delta = np.dot(layer_2_delta, weights[1].T) * relu_deriv(layers[1])

        weights[1] += np.dot(layers[1].T, layer_2_delta) * alpha
        weights[0] += np.dot(layers[0].T, layer_1_delta) * alpha


for data in product([0, 1, 2], repeat=3):
    data = np.array(data, dtype=np.float32).reshape(1, 3)
    expect = int(data[0, 0] != data[0, 1])

    output = nn(data, *weights)[-1]
    print(f"Output: {output[0, 0]} Expected: {expect}")
        

