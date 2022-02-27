import numpy as np
# import tensorflow as tf
import random


def generate_numbers(alphabet: int, alphabet_length: int, prefix=None, arrays=None):
    """
        Генерирует все возможные варианты из алфавита N длиной M.
        prefix - список, длины М
    """
    if arrays is None:
        arrays = []
    prefix = prefix or []
    if alphabet_length == 0:  # крайний случай, добавляем полученную комбинацию в список arrays
        arrays.append(prefix[:])
        return
    for digit in range(alphabet):  # иначе, берем цифры из алфавита N
        prefix.append(digit)  # добавляем цифру к prefix'y
        generate_numbers(alphabet, alphabet_length - 1, prefix, arrays)  # вызываем функцию generate_numbers
        prefix.pop()  # удаляем последние значение в prefix, чтобы добавить другую цифру алфавита
    return arrays


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


def trellis(matrix_of_polynomials):
    number_of_registers = matrix_of_polynomials[0].size - 1  # Количество регистров
    matrix_state = {}
    matrix_transition = {}
    check_transition = {}
    for i in range(2 ** number_of_registers // 2):  # Заполнение матрицы состояний
        matrix_state[2 * i] = (i, i + 2 ** number_of_registers // 2)
        matrix_state[2 * i + 1] = (i, i + 2 ** number_of_registers // 2)
        matrix_transition[2 * i] = []
        matrix_transition[2 * i + 1] = []
        check_transition[2 * i] = [False, False]
        check_transition[2 * i + 1] = [False, False]
    bits = np.array([0, 1])  # Биты для заполнения матрицы переходов
    states = generate_numbers(2, number_of_registers)  # Генерация всех состояний
    column = 0
    for state in states:
        for bit in bits:  # Заполнение матрицы переходов
            matrix_transition[column // 2].append(coding_with_speed_1_2(np.array([bit]), matrix_of_polynomials, state))
            column += 1
    return matrix_state, matrix_transition, check_transition


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


def check_next_state(state_matrix, state, check_transition):
    if check_transition[state][1]:
        bit = 0
        new_state = state_matrix[state][bit]
        check_transition[state][bit] = True
        return new_state
    if check_transition[state][0]:
        bit = 1
        new_state = state_matrix[state][bit]
        check_transition[state][bit] = True
        return new_state
    if sum(check_transition[state_matrix[state][0]]) == sum(check_transition[state_matrix[state][1]]):
        bit = np.random.randint(2)
        new_state = state_matrix[state][bit]
        check_transition[state][bit] = True
        return new_state
    if sum(check_transition[state_matrix[state][0]]) < sum(check_transition[state_matrix[state][1]]):
        bit = 0
        new_state = state_matrix[state][bit]
        check_transition[state][bit] = True
        return new_state
    if sum(check_transition[state_matrix[state][0]]) > sum(check_transition[state_matrix[state][1]]):
        bit = 1
        new_state = state_matrix[state][bit]
        check_transition[state][bit] = True
        return new_state


def choice_next_state(state_matrix, state, check_transition):
    if state == state_matrix[state][0] and not check_transition[state][0]:
        check_transition[state][0] = True
        return state
    if state == state_matrix[state][1] and not check_transition[state][1]:
        check_transition[state][1] = True
        return state
    return check_next_state(state_matrix, state, check_transition)


def check_transition_zeroing(check_transition):
    for i in range(len(check_transition)):
        check_transition[i] = [False, False]


def train_all_data(matrix_of_polynomials, iteration):
    state_matrix, transition_matrix, check_transition = trellis(matrix_of_polynomials)
    number_of_registers = matrix_of_polynomials[0].size  # Количество регистров
    bits = np.zeros(((2 ** number_of_registers + 1) * iteration), dtype=int)
    out_bits = np.zeros((bits.size, 2), dtype=int)
    state = np.random.randint(2 ** (number_of_registers - 1))
    counter = 0
    for i in range(iteration):
        for j in range(2 ** number_of_registers):
            new_state = choice_next_state(state_matrix, state, check_transition)
            if state_matrix[state][0] == new_state:
                bits[counter] = 0
                out_bits[counter] = transition_matrix[state][0]
                counter += 1
            else:
                bits[counter] = 1
                out_bits[counter] = transition_matrix[state][1]
                counter += 1
            state = new_state
        bit = np.random.randint(2)
        new_state = state_matrix[state][bit]
        if state_matrix[state][0] == new_state:
            bits[counter] = 0
            out_bits[counter] = transition_matrix[state][0]
            counter += 1
        else:
            bits[counter] = 1
            out_bits[counter] = transition_matrix[state][1]
            counter += 1
        state = new_state
        check_transition_zeroing(check_transition)
    return out_bits

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


def choice_hyperparameters(labels, model):
    optimizers = np.array(['SGD', 'RMSprop', 'Adam', 'RMSprop', 'Adamax', 'Adadelta', 'Nadam'])
    metrics = np.array(['Accuracy'])
    loss = np.array(['MeanSquaredError'])
    epochs = np.arange(100, 300, 15, dtype=int)
    batch_size = np.arange(18, 40, random.randint(1, 4), dtype=int)
    layers(labels, model)
    return random.choice(optimizers), random.choice(metrics), random.choice(loss), random.choice(epochs), random.choice(
        batch_size)


def neural_model(hyperparameters, model, x_train, y_train, x_test, y_test):
    optimizer = hyperparameters[0]
    metric = hyperparameters[1]
    loss = hyperparameters[2]
    epochs = hyperparameters[3]
    batch_size = hyperparameters[4]
    model.compile(optimizer=optimizer, loss=loss, metrics=metric)
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)
    result_test = model.evaluate(x_test, y_test)
    return result_test


def layers(labels, model):
    layers_count = random.choice(np.arange(2, 4, 1, dtype=int))
    layer_size = np.arange(30, 70, random.randint(1, 4), dtype=int)
    activation = np.array(['relu', 'elu', 'selu', 'tanh', 'softsign'])
    model.add(tf.keras.layers.Dense(random.choice(layer_size), input_dim=labels, activation=random.choice(activation)))
    for i in range(layers_count - 1):
        model.add(tf.keras.layers.Dense(units=random.choice(layer_size), activation=random.choice(activation)))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))


def choice_neural_model(labels, model, iterations, bits_size, matrix_of_polynomials):
    x_train, y_train = train_all_data(matrix_of_polynomials)
    print(x_train, y_train)
    x_test, y_test = data_training(bits_size, labels, matrix_of_polynomials)
    hyperparameters_result = []
    test_result = []
    for _ in range(iterations):
        hyperparameters = choice_hyperparameters(labels, model)
        print(hyperparameters)
        hyperparameters_result.append(hyperparameters)
        test_result.append(neural_model(hyperparameters, model, x_train, y_train, x_test, y_test))
    max_accuracy = test_result[0][1]
    min_loss = test_result[0][0]
    max_param = hyperparameters_result[0]
    for param, result in zip(hyperparameters_result, test_result):
        if result[1] > max_accuracy and result[0] < min_loss:
            max_accuracy = result[1]
            max_param = param
            min_loss = result[0]
    print(min_loss, max_accuracy)


# g1 = np.array([1, 1, 1])
# g2 = np.array([1, 0, 1])
# g1 = np.array([1, 0, 1, 1, 0, 1, 1])  # 133
# g2 = np.array([1, 1, 1, 1, 0, 0, 1])  # 171
g1 = np.array([1, 1, 0, 1])  # 15
g2 = np.array([1, 1, 1, 1])  # 17
matrix_of_polynomials = np.array([g1, g2])
bits_size = 5000
window = 16
iterations = 2
# model = tf.keras.models.Sequential()
# state_matrix, transition_matrix, check_transition = trellis(matrix_of_polynomials)
# print(state_matrix, transition_matrix, check_transition, sep='\n')
print(train_all_data(matrix_of_polynomials, iterations))
# choice_neural_model(window, model, iterations, bits_size, matrix_of_polynomials)
