import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt
from datetime import datetime


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


def get_states(matrix_of_polynomials, code_sequences):
    state_matrix, transition_matrix, _ = trellis(matrix_of_polynomials)
    states = np.zeros(code_sequences.size // 2, dtype=int)
    for i in range(code_sequences.size // 2 - 1):
        if np.all(transition_matrix[states[i]][0] == code_sequences[2 * i: 2 * i + 2]):
            states[i + 1] = state_matrix[states[i]][0]
        else:
            states[i + 1] = state_matrix[states[i]][1]
    return states[1:]


def data_training_random(matrix_of_polynomials, bits_size, step):
    err_bits = step // 2 - 1
    bits_size += err_bits
    out_bits = np.random.randint(2, size=bits_size)
    code_sequences = coding_with_speed_1_2(out_bits, matrix_of_polynomials)
    states = get_states(matrix_of_polynomials, code_sequences)
    get_states(matrix_of_polynomials, code_sequences)
    x_train = np.zeros((bits_size - err_bits, step), np.int32)
    y_train = out_bits[:bits_size - err_bits]
    start = 0
    for i in range(bits_size - err_bits):
        x_train[i] = np.array(code_sequences[start:step])
        start += 2
        step += 2
    return x_train, y_train.T, states[:y_train.size]


def data_test_with_error(matrix_of_polynomials, bits_size, step, err):
    err_bits = step // 2 - 1
    bits_size += err_bits
    out_bits = np.random.randint(2, size=bits_size)
    code_sequences = coding_with_speed_1_2(out_bits, matrix_of_polynomials)
    code_sequences_with_noise = add_noise_in_bits(code_sequences, err)
    start = 0
    x_train = np.zeros((bits_size - err_bits, step), np.int32)
    y_train = out_bits[:bits_size - err_bits]
    for i in range(bits_size - err_bits):
        x_train[i] = np.array(code_sequences_with_noise[start:step])
        start += 2
        step += 2
    return x_train, y_train.T


def choice_next_state_with_repetition(state_matrix, state, check_transition):
    if state == state_matrix[state][0] and not check_transition[state][0]:
        check_transition[state][0] = True
        return state
    if state == state_matrix[state][1] and not check_transition[state][1]:
        check_transition[state][1] = True
        return state
    if sum(check_transition[state]) == 2 and state == state_matrix[state][0]:
        bit = 1
        new_state = state_matrix[state][bit]
        return new_state
    if sum(check_transition[state]) == 2 and state == state_matrix[state][1]:
        bit = 0
        new_state = state_matrix[state][bit]
        return new_state
    if sum(check_transition[state]) == 2:
        bit = np.random.randint(2)
        new_state = state_matrix[state][bit]
        return new_state
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

    else:
        bit = np.random.randint(2)
        new_state = state_matrix[state][bit]
        check_transition[state][bit] = True
        return new_state


def check_transition_sum(check_transition):
    result = 0
    for i in range(len(check_transition)):
        result += sum(check_transition[i])
    return result


def data_training_with_repetition(matrix_of_polynomials, bits_size, step):
    state_matrix, transition_matrix, check_transition = trellis(matrix_of_polynomials)
    number_of_registers = matrix_of_polynomials[0].size  # Количество регистров
    bits = []
    code_sequences = []
    state = np.random.randint(2 ** (number_of_registers - 1))
    states = []
    while len(bits) < bits_size:
        while not (check_transition_sum(check_transition) == 2 ** number_of_registers):
            states.append(state)
            new_state = choice_next_state_with_repetition(state_matrix, state, check_transition)
            if state_matrix[state][0] == new_state:
                bits.append(0)
                code_sequences.append(transition_matrix[state][0])
            else:
                bits.append(1)
                code_sequences.append(transition_matrix[state][1])
            state = new_state
        bit = np.random.randint(2)
        new_state = state_matrix[state][bit]
        state = new_state
        check_transition_zeroing(check_transition)
    err_bits = step // 2 - 1
    states = np.array(states[:bits_size])
    x_train = np.zeros((bits_size, step), np.int32)
    y_train = np.array(bits[:bits_size])
    code_sequences = np.array(code_sequences[:bits_size + err_bits]).reshape((bits_size + err_bits) * 2)
    start = 0
    for i in range(bits_size):
        x_train[i] = code_sequences[start:step]
        start += 2
        step += 2
    return x_train, y_train


def choice_next_state_without_repetition(state_matrix, state, check_transition):
    if state == state_matrix[state][0] and not check_transition[state][0]:
        check_transition[state][0] = True
        return state
    if state == state_matrix[state][1] and not check_transition[state][1]:
        check_transition[state][1] = True
        return state
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
    else:
        bit = np.random.randint(2)
        new_state = state_matrix[state][bit]
        check_transition[state][bit] = True
        return new_state


def check_transition_zeroing(check_transition):
    for i in range(len(check_transition)):
        check_transition[i] = [False, False]


def plot_states_without_repetition(*states):
    numbers_of_states = np.arange(0, len(states[0]), dtype=int)
    fig, ax = plt.subplots()
    for state in states:
        ax.plot(numbers_of_states, state)
    ax.grid()
    # ax.legend()
    ax.invert_yaxis()
    plt.show()


def find_final_states(final_states, state, state_matrix):
    counter = 0
    for key, value in state_matrix.items():
        if state == value[0]:
            final_states[counter] = key
            counter += 1
            final_states[-1] = 0
        elif state == value[1]:
            final_states[counter] = key
            counter += 1
            final_states[-1] = 1


def data_training_without_repetition(matrix_of_polynomials, bits_size, step):
    state_matrix, transition_matrix, check_transition = trellis(matrix_of_polynomials)
    number_of_registers = matrix_of_polynomials[0].size  # Количество регистров
    bits = []
    code_sequences = []
    state = np.random.randint(2 ** (number_of_registers - 1))
    states = []
    while len(bits) < bits_size:
        while not (sum(check_transition[state]) == 2):
            states.append(state)
            new_state = choice_next_state_without_repetition(state_matrix, state, check_transition)
            if state_matrix[state][0] == new_state:
                bits.append(0)
                code_sequences.append(transition_matrix[state][0])
            else:
                bits.append(1)
                code_sequences.append(transition_matrix[state][1])
            state = new_state
        bit = np.random.randint(2)
        new_state = state_matrix[state][bit]
        state = new_state
        check_transition_zeroing(check_transition)
    states = np.array(states[:bits_size])
    err_bits = step // 2 - 1
    x_train = np.zeros((bits_size, step), np.int32)
    y_train = np.array(bits[:bits_size])
    code_sequences = np.array(code_sequences[:bits_size + err_bits]).reshape((bits_size + err_bits) * 2)
    start = 0
    for i in range(bits_size):
        x_train[i] = code_sequences[start:step]
        start += 2
        step += 2
    return x_train, y_train, states


def choice_hyperparameters(labels, model):
    optimizers = np.array(['SGD', 'RMSprop', 'Adam', 'RMSprop', 'Adamax', 'Adadelta', 'Nadam'])
    metrics = np.array(['Accuracy'])
    loss = np.array(['MeanSquaredError'])
    epochs = np.arange(100, 300, dtype=int)
    batch_size = np.arange(18, 40, dtype=int)
    layers_and_activation = layers(labels, model)
    return random.choice(optimizers), random.choice(metrics), random.choice(loss), random.choice(epochs), random.choice(
        batch_size), layers_and_activation


def neural_model(hyperparameters, model, x_train, y_train, x_test, y_test):
    optimizer = hyperparameters[0]
    metric = hyperparameters[1]
    loss = hyperparameters[2]
    epochs = hyperparameters[3]
    batch_size = hyperparameters[4]
    model.compile(optimizer=optimizer, loss=loss, metrics=metric)
    history = model.fit(x_train, y_train, validation_split=0.15, epochs=epochs, batch_size=batch_size, verbose=0)
    result_test = model.evaluate(x_test, y_test)
    return history, result_test


def layers(labels, model):
    layers_count = random.choice(np.arange(2, 4, 1, dtype=int))
    layers_and_activation = {}
    layer_size = np.arange(30, 70, dtype=int)
    activations = np.array(['relu', 'elu', 'selu', 'tanh', 'softsign'])
    units = random.choice(layer_size)
    activation = random.choice(activations)
    model.add(tf.keras.layers.Dense(units, input_dim=labels, activation=activation))
    layers_and_activation[0] = (units, activation)
    for i in range(layers_count - 1):
        units = random.choice(layer_size)
        activation = random.choice(activations)
        model.add(tf.keras.layers.Dense(units=units, activation=activation))
        layers_and_activation[i + 1] = (units, activation)
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    layers_and_activation[layers_count] = (1, 'sigmoid')
    return layers_and_activation


def choice_neural_model(labels, iterations, bits_size, matrix_of_polynomials):
    x_train_with_repetition, y_train_with_repetition, _ = data_training_with_repetition(matrix_of_polynomials,
                                                                                        bits_size, labels)
    x_train_without_repetition, y_train_without_repetition, _ = data_training_without_repetition(matrix_of_polynomials,
                                                                                                 bits_size, labels)
    x_train_rand, y_train_rand, _ = data_training_random(matrix_of_polynomials, bits_size, labels)
    x_train = [x_train_rand, x_train_with_repetition, x_train_without_repetition]
    y_train = [y_train_rand, y_train_with_repetition, y_train_without_repetition]
    x_test, y_test, _ = data_training_random(matrix_of_polynomials, bits_size, labels)
    the_best_neural_models = []
    for i in range(3):
        stories = []
        results = []
        hyperparameters = []
        for _ in range(iterations):
            model = tf.keras.models.Sequential()
            hyperparameter = choice_hyperparameters(labels, model)
            hyperparameters.append(hyperparameter)
            model.summary()
            history, result = neural_model(hyperparameter, model, x_train[i], y_train[i], x_test, y_test)
            stories.append(history)
            results.append(result[0])
            tf.keras.backend.clear_session()
        the_best_neural_models.append(hyperparameters[choice_the_best_neural_model(results)])
    return the_best_neural_models


def write_the_best_neural_models(window, bits, the_best_neural_models):
    path = 'model created: ' + str(datetime.now())
    data = open(path, "w")
    np.savetxt(data, np.array([window]))
    np.savetxt(data, np.array([bits]))
    np.savetxt(data, the_best_neural_models, fmt="%s")
    data.close()


def plot_training(stories):
    for history in stories:
        fig, ax = plt.subplots()
        ax.plot(history.history['loss'])
        ax.plot(history.history['val_loss'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()


def choice_the_best_neural_model(result):
    min_value = min(result)
    min_index = result.index(min_value)
    return min_index


def neural_network(matrix_of_polynomials, bits_size, labels):
    x_train, y_train = data_training_with_repetition(matrix_of_polynomials, bits_size, labels)
    x_test, y_test, = data_test_with_error(matrix_of_polynomials, bits_size, labels, 0.0)
    input = tf.keras.Input(shape=(labels,), name='input')
    layer_1 = tf.keras.layers.Dense(units=59, activation='elu', name='hidden_layer_1')(input)
    layer_2 = tf.keras.layers.Dense(units=51, activation='tanh', name='hidden_layer_2')(layer_1)
    layer_3 = tf.keras.layers.Dense(units=33, activation='relu', name='hidden_layer_3')(layer_2)
    output = tf.keras.layers.Dense(units=1, activation='sigmoid', name='output')(layer_3)
    model = tf.keras.Model(inputs=input, outputs=output)
    model.compile(optimizer='NAdam', loss=['MeanSquaredError'], metrics=['accuracy'])
    history = model.fit(x_train, y_train, validation_split=0.15, epochs=190, batch_size=30, verbose=1)
    results = model.evaluate(x_test, y_test)
    predictions = model.predict(x_test)
    for x, y in zip(y_test, predictions):
        print(x, y)
    predictions = np.where(predictions > 0.5, 1, 0)
    print(signal_comparison(y_test, predictions))
    print(results)
    # plot_training(stories)
    return 0


def signal_comparison(out_bits: np.array, in_bits: np.array):
    out_bits = out_bits.reshape(in_bits.shape[0], in_bits.shape[1])
    errors = sum((out_bits + in_bits) % 2)
    return errors


# g1 = np.array([1, 1, 1])
# g2 = np.array([1, 0, 1])
# g1 = np.array([1, 1, 0, 1])  # 15
# g2 = np.array([1, 1, 1, 1])  # 17
# g1 = np.array([1, 0, 0, 1, 1])  # 23
# g2 = np.array([1, 1, 1, 0, 1])  # 35
g1 = np.array([1, 0, 1, 1, 0, 1, 1])  # 133
g2 = np.array([1, 1, 1, 1, 0, 0, 1])  # 171

matrix_of_polynomials = np.array([g1, g2])
bits_size = 5000
window = 18
iterations = 10
neural_network(matrix_of_polynomials, bits_size, window)
