import numpy as np
# import sys
from keras.datasets import mnist


ALPHA = 0.005
HIDDEN_SIZE = 40
PIXELS_PER_IMAGE = INPUT_SIZE = 28*28
OUTPUT_SIZE = 10
TRAIN_DATA_SIZE = 1000
LAYERS_TOTAL = 3


(x_train, y_train), (x_test, y_test) = mnist.load_data()

images, labels = x_train[0:TRAIN_DATA_SIZE] / 255, y_train[0:TRAIN_DATA_SIZE]


data_to_learn = images.reshape(TRAIN_DATA_SIZE, PIXELS_PER_IMAGE)
outputs_to_learn = np.zeros((TRAIN_DATA_SIZE, OUTPUT_SIZE))

for ind, label in enumerate(labels):
    outputs_to_learn[ind][label] = 1.0


test_data = x_test.reshape(len(x_test), PIXELS_PER_IMAGE) / 255
test_outputs = np.zeros((len(y_test), OUTPUT_SIZE))

for ind, label in enumerate(y_test):
    test_outputs[ind][label] = 1.0


np.random.seed(1)
def relu(x): return (x > 0) * x
def relu_deriv(x): return x > 0

weights = [
    0.2 * np.random.rand(INPUT_SIZE, HIDDEN_SIZE) - 0.1,
    0.2 * np.random.rand(HIDDEN_SIZE, OUTPUT_SIZE) - 0.1,
]

def nn(data, weights):
    global LAYERS_TOTAL
    layers = [None] * LAYERS_TOTAL
    data = data_to_learn[j].reshape(1, INPUT_SIZE)

    layers[0] = data
    layers[1] = relu(np.dot(layers[0], weights[0]))
    layers[2] = np.dot(layers[1], weights[1])

    return layers


for i in range(350):
    for j in range(TRAIN_DATA_SIZE):
        # Output
        data = data_to_learn[j].reshape(1, INPUT_SIZE)
        layers = nn(data=data, weights=weights)

        # Learning
        layer_deltas = [None] * LAYERS_TOTAL
        expected_output = outputs_to_learn[j].reshape(1, OUTPUT_SIZE)
        layer_deltas[2] = expected_output - layers[2]
        layer_deltas[1] = np.dot(layer_deltas[2], weights[1].T) * relu_deriv(layers[1])

        weights[1] += np.dot(layers[1].T, layer_deltas[2]) * ALPHA
        weights[0] += np.dot(layers[0].T, layer_deltas[1]) * ALPHA


correct_counter = 0
for i in range(len(test_data)):
    data = test_data[i].reshape(1, INPUT_SIZE)
    layers = nn(data=data, weights=weights)

    expected_output = test_outputs[i].reshape(1, OUTPUT_SIZE)
    correct_counter += round(np.argmax(expected_output)) == round(np.argmax(layers[2]))

print(f"Correct: {correct_counter / len(test_outputs)}")


