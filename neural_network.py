import numpy as np
import tensorflow as tf
import random


def coding_with_speed_1_2(bits: np.array, matrix_of_polynomials: np.array, states=None):
    if states:  # Для построения матриц состояний и переходов
        states = np.flip(states)  # Инверсия состояний
        bits = np.insert(bits, 0, states)  # Добавление нулей в начало массива (состояния регистра)
    else:
        bits = np.insert(bits, 0,
                         np.zeros(
                             matrix_of_polynomials[
                                 0].size - 1))  # Добавление нулей в начало массива (состояния регистра)
        # bits = np.append(bits, np.zeros(matrix_of_polynomials[0].size - 1,
        #                                 dtype='int'))  # Добавление нулей в конец массива (с "хвостом")
    coding_sequences = np.zeros((bits.size - matrix_of_polynomials[0].size + 1) * 2,
                                dtype='int')  # Кодовая последовательность
    count = 0  # Счётчик для кодовой последовательности
    for k in range(matrix_of_polynomials[0].size - 1, bits.size):
        for polynomial in matrix_of_polynomials:
            for j in range(polynomial.size):
                coding_sequences[count] += (
                        bits[k - j] * polynomial[j])  # Умножение порождающего полинома на информационные биты
            coding_sequences[count] = coding_sequences[count] % 2
            count += 1
    return coding_sequences


def add_noise_in_bits(bits: np.array, bit_error_rate: float):
    bits_with_error = np.copy(bits)
    for i in range(bits_with_error.size):
        if np.random.uniform(0, 1) < bit_error_rate:
            bits_with_error[i] = (~bits_with_error[i]) % 2
    return bits_with_error


def data_training(bits_size, step, matrix_of_polynomials):
    err_bits = step // 2 - 1
    bits_size += err_bits
    matrix_of_polynomials = np.array([g1, g2])
    out_bits = np.random.randint(2, size=bits_size)
    code_sequences = coding_with_speed_1_2(out_bits, matrix_of_polynomials)
    start = 0
    x_train = np.zeros((bits_size - err_bits, step), np.int32)
    y_train = out_bits[:bits_size - err_bits]
    for i in range(bits_size - err_bits):
        x_train[i] = np.array(code_sequences[start:step])
        start += 2
        step += 2
    return x_train, y_train.T


def data_test_with_error(bits_size, err, step, matrix_of_polynomials):
    err_bits = step // 2 - 1
    bits_size += err_bits
    out_bits = np.random.randint(2, size=bits_size)
    code_sequences = coding_with_speed_1_2(out_bits, matrix_of_polynomials)
    start = 0
    x_train = np.zeros((bits_size - err_bits, step), np.int32)
    y_train = out_bits[:bits_size - err_bits]
    for i in range(bits_size - err_bits):
        x_train[i] = np.array(add_noise_in_bits(code_sequences[start:step], err[i]))
        start += 2
        step += 2
    return x_train, y_train.T


def choice_hyperparameters():
    optimizers = np.array(['SGD', 'RMSprop', 'Adam', 'RMSprop', 'Adamax', 'Adadelta', 'Nadam'])
    metrics = np.array(['Accuracy'])
    loss = np.array(['MeanSquaredError'])
    epochs = np.arange(100, 300, 15, dtype=int)
    batch_size = np.arange(18, 40, random.randint(1, 4), dtype=int)
    layer_1 = np.arange(40, 60, random.randint(1, 4), dtype=int)
    layer_2 = np.arange(50, 70, random.randint(1, 4), dtype=int)
    layer_3 = np.arange(20, 30, random.randint(1, 4), dtype=int)
    activation_1 = np.array(['relu', 'elu', 'selu', 'tanh', 'softsign'])
    activation_2 = np.array(['relu', 'elu', 'selu', 'tanh'])
    activation_3 = np.array(['relu', 'elu', 'selu', 'tanh', 'softsign'])
    activation_output = np.array(['sigmoid'])
    return random.choice(optimizers), random.choice(metrics), random.choice(loss), random.choice(epochs), random.choice(
        batch_size), random.choice(layer_1), random.choice(layer_2), random.choice(layer_3), random.choice(
        activation_1), random.choice(activation_2), random.choice(activation_3), random.choice(activation_output)


def neural_model(hyperparameters, window, x_train, y_train, x_test, y_test):
    optimizers = hyperparameters[0]
    metrics = hyperparameters[1]
    loss = hyperparameters[2]
    epochs = hyperparameters[3]
    batch_size = hyperparameters[4]
    layer_1_size = hyperparameters[5]
    layer_2_size = hyperparameters[6]
    layer_3_size = hyperparameters[7]
    activation_1 = hyperparameters[8]
    activation_2 = hyperparameters[9]
    activation_3 = hyperparameters[10]
    activation_output = hyperparameters[11]
    input = tf.keras.Input(shape=(window,), name='input')
    layer_1 = tf.keras.layers.Dense(units=layer_1_size, activation=activation_1, name='hidden_layer_1')(input)
    layer_2 = tf.keras.layers.Dense(units=layer_2_size, activation=activation_2, name='hidden_layer_2')(layer_1)
    layer_3 = tf.keras.layers.Dense(units=layer_3_size, activation=activation_3, name='hidden_layer_3')(layer_2)
    output = tf.keras.layers.Dense(units=1, activation=activation_output, name='output')(layer_3)
    model = tf.keras.Model(inputs=input, outputs=output)
    model.compile(optimizer=optimizers, loss=loss, metrics=metrics)
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)
    result_test = model.evaluate(x_test, y_test)
    return result_test


def print_hyperparameters(hyperparameters):
    param = ['optimizers', 'metrics', 'loss', 'epochs', 'batch_size', 'layer_1_size', 'layer_2_size', 'layer_3_size',
             'activation_1', 'activation_2', 'activation_3', 'activation_output']
    for x, y in zip(param, hyperparameters):
        print(x, y, sep=': ')


# g1 = np.array([1, 1, 1])
# g2 = np.array([1, 0, 1])
g1 = np.array([1, 0, 1, 1, 0, 1, 1])  # 133
g2 = np.array([1, 1, 1, 1, 0, 0, 1])  # 171
matrix_of_polynomials = np.array([g1, g2])
bits_size = 10000
window = 18
x_train, y_train = data_training(bits_size, window, matrix_of_polynomials)
x_test, y_test = data_training(bits_size, window, matrix_of_polynomials)
hyperparameters_result = []
test_result = []

for _ in range(10):
    hyperparameters = choice_hyperparameters()
    hyperparameters_result.append(hyperparameters)
    test_result.append(neural_model(hyperparameters, window, x_train, y_train, x_test, y_test))
max_accuracy = test_result[0][1]
min_loss = test_result[0][0]
max_param = hyperparameters_result[0]
for param, result in zip(hyperparameters_result, test_result):
    if result[1] > max_accuracy and result[0] < min_loss:
        max_accuracy = result[1]
        max_param = param
        min_loss = result[0]

print(min_loss, max_accuracy)
print_hyperparameters(max_param)
