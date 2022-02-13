import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


size = 100 - 5
traning_inputs = np.loadtxt("traning_data.txt").reshape(size, 12)
traning_inputs = traning_inputs.astype(int)
# print(traning_inputs)
traning_outputs = np.array([np.loadtxt("ans_data.txt")]).T
traning_outputs = traning_outputs.astype(int)
# print(traning_outputs)
synaptic_weights = 2 * np.random.random((12, 1)) - 1
for _ in range(2000):
    input_lauer = traning_inputs
    outputs = sigmoid(np.dot(input_lauer, synaptic_weights))
    err = traning_outputs - outputs
    adjustments = np.dot(input_lauer.T, err * (outputs * (1 - outputs)))
    synaptic_weights += adjustments

# print("Веса после обучения")
# print(synaptic_weights)
# print("Результат")
# print(outputs)
#
# new_inputs = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
# new_inputs = np.array([1, 1, 1, 1, 1, 1, 1, 1])
new_inputs = np.array([1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0])
# new_inputs = np.random.randint(2, size=12)
print(new_inputs)
output = sigmoid(np.dot(new_inputs, synaptic_weights))
print("Новая ситуация")
print(output)
