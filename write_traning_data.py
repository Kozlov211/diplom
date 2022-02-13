import random

import numpy as np


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


np.random.seed(1)
g1 = np.array([1, 1, 1])
g2 = np.array([1, 0, 1])
matrix_of_polynomials = np.array([g1, g2])
bits_size = 1005
out_bits = np.random.randint(2, size=bits_size)
code_sequences = coding_with_speed_1_2(out_bits, matrix_of_polynomials)
# print(out_bits)
# print(code_sequences)
fout_data = open("traning_data.txt", "w")
fout_ans = open("ans_data.txt", "w")
start = 0
step = 12
for i in range(bits_size - 5):
    code = code_sequences[start:step]
    start += 2
    step += 2
    print(code)
    # print(out_bits[0])
    np.savetxt(fout_data, code)
np.savetxt(fout_ans, out_bits[:bits_size - 5])

# print(out_bits[:7])
fout_data.close()
fout_ans.close()
