import numpy as np
import itertools
import cmath
import math
from scipy.spatial import distance
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)


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
        bits = np.append(bits, np.zeros(matrix_of_polynomials[0].size - 1,
                                        dtype='int'))  # Добавление нулей в конец массива (с "хвостом")
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


def coding_with_speed_2_3(bits: np.array, matrix_of_polynomials: np.array):
    # """
    # Кодирует информационную битовую последовательность
    # в соответствии с алгоритмом сверточного кода (пораждающих полиномов), скорость кодера 2/3,
    # возвращает кодовую последовательность.
    # :param bits: [1, 0, 0, 1, 1, 0]
    # :param matrix_of_polynomials: [[1,1], [1, 0, 1]]
    # :return: [1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0]
    # """
    bits = np.insert(bits, 0,
                     np.zeros(matrix_of_polynomials[0][
                                  0].size * 2))  # Добавление нулей в начало массива (состояния регистра)
    bits = np.append(bits, np.zeros((matrix_of_polynomials[0][0].size - 1) * 2,
                                    dtype='int'))  # Добавление нулей в конец массива (с "хвостом")
    coding_sequences = np.zeros((bits.size - matrix_of_polynomials[0][0].size * 2) * 3,
                                dtype='int')  # Кодовая последовательность
    count = 0  # Счётчик для кодовой последовательности
    print(bits)
    for k in range(matrix_of_polynomials[0][0].size, bits.size):
        print(bits[k])
        for polynomials in matrix_of_polynomials:
            var = 0
            for polynomial in polynomials:
                # print(polynomial)
                for j in range(polynomial.size):
                    var += bits[k - j] * polynomial[j]
            # print(var % 2)
            # print(var % 2, count)
            # count += 1
            #     coding_sequences[count] += (
            #             bits[k - j] * polynomial[j])  # Умножение порождающего полинома на информационные биты
            # coding_sequences[count] = coding_sequences[count] % 2
            # count += 1
    return coding_sequences


def trellis(matrix_of_polynomials: np.array):
    number_of_registers = matrix_of_polynomials[0].size - 1  # Количество регистров
    state_matrix = np.zeros([2 ** number_of_registers, 2], dtype='int32')  # Матрица состояний
    transition_matrix = np.zeros([2 ** number_of_registers * 2, 2], dtype='int32')  # Матрица переходов
    for i in range(state_matrix.shape[0] // 2):  # Заполнение матрицы состояний
        state_matrix[2 * i][0] = i
        state_matrix[2 * i][1] = i + state_matrix.shape[0] // 2
        state_matrix[2 * i + 1][0] = i
        state_matrix[2 * i + 1][1] = i + state_matrix.shape[0] // 2
    bits = np.array([0, 1])  # Биты для заполнения матрицы переходов
    states = generate_numbers(2, number_of_registers)  # Генерация всех состояний
    # print(states)
    column = 0
    for bit in bits:  # Заполнение матрицы переходов
        for state in states:
            transition_matrix[column] = coding_with_speed_1_2(np.array([bit]), matrix_of_polynomials, state)
            column += 1
        # print()
    return state_matrix, transition_matrix


def decoding_viterbi(code_sequences: np.array, matrix_of_polynomials: np.array, state_matrix: np.array,
                     transition_matrix: np.array):
    weight = np.full((state_matrix.shape[0], code_sequences.size // 2 + 1), float('inf'))  # Матрица весов состояний
    bits = np.zeros(code_sequences.size // 2, dtype='int')  # Информационные биты
    states = np.zeros(code_sequences.size // 2 + 1, dtype='int')  # Состояния декодера
    weight[0][0] = 0  # Инициализация начального состояния
    for j in range(bits.size):  # Нахождение весов для каждого состояния
        var = 0  # Выбор путей, которые ведут в следующее состояние
        for k in range(state_matrix.shape[0]):
            states_old = np.argwhere(
                state_matrix == k)  # Нахождение состояний, из которых возможно перейти в следующее состояние

            weight[k][j + 1] = min(weight[states_old[0][0]][j] + distance.hamming(code_sequences[2 * j: 2 * j + 2],
                                                                                  transition_matrix[var]) * 2,
                                   weight[states_old[1][0]][j] + distance.hamming(code_sequences[2 * j: 2 * j + 2],
                                                                                  transition_matrix[var + 1]) * 2)
            var += 2
    states[-1] = np.argmin(weight, axis=0)[-1]
    for i in range(code_sequences.size // 2, 0, -1):  # Нахождение состояния декодера
        states_old = np.argwhere(
            state_matrix == states[i])  # Нахождение состояний, из которых возможно перейти в следующее состояние
        min_weight = min(weight[states_old[0][0]][i - 1], weight[states_old[1][0]][i - 1])  # Выбор минимального веса
        if min_weight == weight[states_old[0][0]][i - 1]:
            states[i - 1] = states_old[0][0]
        else:
            states[i - 1] = states_old[1][0]
    for i in range(states.size - 1):  # Декодирование по состояниям декодера
        if state_matrix[states[i]][0] == states[i + 1]:
            bits[i] = 0
        else:
            bits[i] = 1
    return bits[:-matrix_of_polynomials[0].size + 1]  # Отрезаем "хвост" от информационной последовательности бит


def decoding_viterbi_with_window(code_sequences: np.array, matrix_of_polynomials: np.array, state_matrix: np.array,
                                 transition_matrix: np.array, window: int):
    weight = np.full((state_matrix.shape[0], window), float('inf'))  # Матрица весов состояний
    bits = np.zeros(code_sequences.size // 2, dtype='int')  # Информационные биты
    weight[0][0] = 0  # Инициализация начального состояния
    states = np.zeros(window, dtype='int')  # Состояния декодера
    counter_window = window - 2
    counter_bits = 0
    for j in range(window - 1):  # Заполнение окна
        choice_of_paths = 0  # Выбор путей, которые ведут в следующее состояние
        for k in range(state_matrix.shape[0]):
            states_old = np.argwhere(
                state_matrix == k)  # Нахождение состояний, из которых возможно перейти в текущее состояние
            weight[k][j + 1] = min(
                weight[states_old[0][0]][j] + distance.hamming(code_sequences[2 * j: 2 * j + 2],
                                                               transition_matrix[choice_of_paths]) * 2,
                weight[states_old[1][0]][j] + distance.hamming(code_sequences[2 * j: 2 * j + 2],
                                                               transition_matrix[choice_of_paths + 1]) * 2)
            choice_of_paths += 2
    states[-1] = np.argmin(weight, axis=0)[-1]  # Выбор минимального веса в последнем столбце матрицы
    for i in range(weight.shape[1] - 1, 0, -1):  # Нахождение состояния декодера
        states_old = np.argwhere(state_matrix == states[i])  # Нахождение состояний, из которых возможно перейти
        min_weight = min(weight[states_old[0][0]][i - 1],
                         weight[states_old[1][0]][i - 1])  # Выбор минимального веса
        if min_weight == weight[states_old[0][0]][i - 1]:  # Выбор состояний декодера
            states[i - 1] = states_old[0][0]
        else:
            states[i - 1] = states_old[1][0]
    if state_matrix[states[0]][0] == states[1]:  # Декодирование первого бита в окне
        bits[counter_bits] = 0
        counter_bits += 1
    else:
        bits[counter_bits] = 1
        counter_bits += 1
    weight[:, :-1], weight[:, -1] = weight[:, 1:], 0  # Удаление первого столбца из матрицы весов
    states[:-1], states[-1] = states[1:], 0  # Удаление первого столбца из состояний декодера
    for j in range(window - 1, bits.size):
        choice_of_paths = 0  # Выбор путей, которые ведут в следующее состояние
        for k in range(state_matrix.shape[0]):
            states_old = np.argwhere(
                state_matrix == k)  # Нахождение состояний, из которых возможно перейти в текущее состояние
            weight[k][counter_window + 1] = min(
                weight[states_old[0][0]][counter_window] + distance.hamming(code_sequences[2 * j: 2 * j + 2],
                                                                            transition_matrix[choice_of_paths]) * 2,
                weight[states_old[1][0]][counter_window] + distance.hamming(code_sequences[2 * j: 2 * j + 2],
                                                                            transition_matrix[choice_of_paths + 1]) * 2)
            choice_of_paths += 2
        states[-1] = np.argmin(weight, axis=0)[-1]  # Выбор минимального веса в последнем столбце матрицы
        if state_matrix[states[0]][0] == states[1]:  # Декодирование первого бита в окне
            bits[counter_bits] = 0
            counter_bits += 1
        else:
            bits[counter_bits] = 1  # Декодирование первого бита в окне
            counter_bits += 1
        weight[:, :-1], weight[:, -1] = weight[:, 1:], 0  # Удаление первого столбца из матрицы весов
        states[:-1], states[-1] = states[1:], 0  # Удаление первого столбца из состояний декодера
    if bits.size - counter_bits > 0:  # Если не декодированы все биты в последнем окне
        for i in range(states.size - 2):  # Декодирование по состояниям декодера
            if state_matrix[states[i]][0] == states[i + 1]:  # Декодирование оставшихся битов
                bits[counter_bits] = 0
                counter_bits += 1
            else:
                bits[counter_bits] = 1  # Декодирование оставшихся битов
                counter_bits += 1
    return bits[:-matrix_of_polynomials[0].size + 1]


def add_noise_in_bits(bits: np.array, bit_error_rate: float):
    bits_with_error = np.copy(bits)
    for i in range(bits_with_error.size):
        if np.random.uniform(0, 1) < bit_error_rate:
            bits_with_error[i] = (~bits_with_error[i]) % 2
    return bits_with_error


def signal_comparison(out_bits: np.array, in_bits: np.array):
    errors = sum((out_bits + in_bits) % 2)
    return errors


def finding_the_probability_of_error(g1, g2):
    out_bits = np.random.randint(2, size=10000)
    matrix_of_polynomials = np.array([g1, g2])
    code_sequences = coding_with_speed_1_2(out_bits, matrix_of_polynomials)
    state_matrix, transition_matrix = trellis(matrix_of_polynomials)
    p1 = np.linspace(0.0001, 0.2, 10)
    p2 = np.zeros(p1.size)
    for i in range(p1.size):
        code_sequences_with_noise = add_noise_in_bits(code_sequences, p1[i])
        in_bits = decoding_viterbi(code_sequences_with_noise, matrix_of_polynomials, state_matrix, transition_matrix)
        p2[i] = signal_comparison(out_bits, in_bits) / out_bits.size
        print(p2[i], i)
    with open('p1.txt', 'w') as file:
        file.write(''.join(np.array2string(p1, separator=',').splitlines()))
    with open('p2.txt', 'w') as file:
        file.write(''.join(np.array2string(p2, separator=',').splitlines()))


# out_bits = np.array([1, 0, 1, 1, 0])
# out_bits = np.ones(10, dtype='int')
out_bits = np.random.randint(2, size=1000)
# !!!СТЕПЕНЬ МНОГОЧЛЕНА СЛЕВА - 0!!!
g1 = np.array([1, 0, 1, 1, 0, 1, 1])  # 133
g2 = np.array([1, 1, 1, 1, 0, 0, 1])  # 171
# g1 = np.array([1, 0, 0, 1, 1])  # 23
# g2 = np.array([1, 1, 1, 0, 1])  # 35
# g1 = np.array([1, 1, 0, 1])  # 15
# g2 = np.array([1, 1, 1, 1])  # 17
# g1 = np.array([1, 1, 1])
# g2 = np.array([1, 0, 1])
matrix_of_polynomials = np.array([g1, g2])
print('Hello')
code_sequences = coding_with_speed_1_2(out_bits, matrix_of_polynomials)
state_matrix, transition_matrix = trellis(matrix_of_polynomials)
# print(state_matrix)
# states_old = np.argwhere(state_matrix == 1)
# print(states_old)
# print(states_old[0][0], states_old[1][0])
# print(transition_matrix[:state_matrix.shape[0]], transition_matrix[state_matrix.shape[0]:], sep='\n')
in_bits = decoding_viterbi(code_sequences, matrix_of_polynomials, state_matrix, transition_matrix)
print(signal_comparison(out_bits, in_bits))
# print(out_bits, in_bits, sep='\n')
# in_bits = decoding_viterbi_with_window(code_sequences, matrix_of_polynomials, state_matrix, transition_matrix, 100)
# print(out_bits, in_bits,sep='\n')
# print(signal_comparison(out_bits, in_bits))
# finding_the_probability_of_error(g1, g2)
# from datetime import datetime
# start_time = datetime.now()
# in_bits = decoding_viterbi_with_window(code_sequences, matrix_of_polynomials, state_matrix, transition_matrix, 1000)
# end_time = datetime.now()
# print(signal_comparison(out_bits, in_bits))
# print('Duration: {}'.format(end_time - start_time))
# start_time = datetime.now()
# in_bits = decoding_viterbi(code_sequences, matrix_of_polynomials, state_matrix, transition_matrix)
# end_time = datetime.now()
# print(signal_comparison(out_bits, in_bits))
# print('Duration: {}'.format(end_time - start_time))
