import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import warnings


def normalize(dataset):
    mu = np.mean(dataset, axis=0)
    sigma = np.std(dataset, axis=0)
    return (dataset - mu) / sigma


def run_logistic_model(learning_r, training_epochs, train_obs, train_labels, debug=False):
    sees = tf.compat.v1.Session()
    sees.run(init)
    cost_history = np.empty(shape=[0], dtype=float)
    for epoch in range(training_epochs + 1):
        sees.run(training_step, feed_dict={X: train_obs, Y: train_labels, learning_rate: learning_r})
        cost_ = sees.run(cost, feed_dict={X: train_obs, Y: train_labels, learning_rate: learning_r})
        cost_history = np.append(cost_history, cost_)
        if (epoch % 1000 == 0) and debug:
            print('Достигнута эпоха', epoch, 'стоимость J=', str.format('{0:0.6f}', cost_))
    return sees, cost_history


size = 1000

tf.compat.v1.disable_eager_execution()
features = np.loadtxt("traning_data.txt").reshape(size, 12)
features = features.astype(int)
labels = np.array(np.loadtxt("ans_data.txt"))  # Этикетка
labels = labels.astype(int)
n_training_samples = features.shape[0]  # Тренировочные данные
n_dim = features.shape[1]  # Признаки
print('Набор данных имеет', n_training_samples, 'тренировочных образцов')
print('Набор данных имеет', n_dim, 'признаков')
features_norm = normalize(features)  # Нормализация признаков
train_x = np.transpose(features_norm)
train_y = labels.reshape(1, len(labels))
X = tf.compat.v1.placeholder(tf.float32, [n_dim, None])  # Содержит матрицу X[xn, m] m - не объявляется
Y = tf.compat.v1.placeholder(tf.float32, [1, None])  # Содержит выходные значения размерности [1, m]
learning_rate = tf.compat.v1.placeholder(tf.float32, shape=())  # Содержит темп заучивания
stddev = 2 / np.sqrt(n_dim)
W = tf.compat.v1.Variable(tf.random.truncated_normal([n_dim, 1], stddev=stddev))  # Содержит матрицу весов [xn, 1]
b = tf.compat.v1.Variable(tf.zeros(1))  # Содержит смещение
init = tf.compat.v1.global_variables_initializer()  # Инициализация переменных
y_ = tf.sigmoid(tf.matmul(tf.transpose(W), X) + b)  # Вычисляет выходной нейрон
cost = - tf.reduce_mean(Y * tf.math.log(y_) + (1 - Y) * tf.math.log(1 - y_))  # Определяет стоимостную функцию
training_step = tf.compat.v1.train.GradientDescentOptimizer(learning_rate).minimize(cost)
sees, cost_history = run_logistic_model(learning_r=0.002, training_epochs=20000, train_obs=train_x,
                                        train_labels=train_y, debug=True)

correct_predictionl = tf.equal(tf.greater(y_, 0.5), tf.equal(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_predictionl, tf.float32))
print(sees.run(accuracy, feed_dict={X: train_x, Y: train_y, learning_rate: 0.002}))
# print(sees.run(W))
sees.close()

plt.plot(cost_history)
plt.grid()
plt.show()
