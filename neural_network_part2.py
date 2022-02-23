import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import warnings


def normalize(dataset):
    mu = np.mean(dataset, axis=0)
    sigma = np.std(dataset, axis=0)
    return (dataset - mu) / sigma


def model(minibatch_size, training_epochs, features, classes, logging_step=100, learning_r=0.001):
    sees = tf.compat.v1.Session()
    sees.run(init)
    cost_history = []
    for epoch in range(training_epochs + 1):
        for i in range(0, features.shape[1], minibatch_size):
            x_train_mini = features[:, i:i + minibatch_size]
            y_train_mini = classes[:, i:i + minibatch_size]
            sees.run(optimizer, feed_dict={X: x_train_mini, Y: y_train_mini, learning_rate: learning_r})
        cost_ = sees.run(cost, feed_dict={X: features, Y: classes, learning_rate: learning_r})
        cost_history = np.append(cost_history, cost_)
        if (epoch % logging_step == 0):
            print('Достигнута эпоха', epoch, 'стоимость J=', str.format('{0:0.6f}', cost_))
    return sees, cost_history


def create_layer(X, n, activation):
    n_dim = int(X.shape[0])
    stddev = 2 / np.sqrt(n_dim)
    initialization = tf.random.truncated_normal((n, n_dim), stddev=stddev)
    W = tf.compat.v1.Variable(initialization)  # Содержит матрицу весов [xn, 1]
    b = tf.compat.v1.Variable(tf.zeros(n, 1))  # Содержит смещение
    z = tf.matmul(W, X) + b
    return activation(z)


size = 1000
tf.compat.v1.disable_eager_execution()
features = np.loadtxt("traning_data.txt").reshape(size, 12)
features = features.astype(int)
labels = np.array(np.loadtxt("ans_data.txt"))  # Этикетка
labels = labels.astype(int)
n_training_samples = features.shape[0]  # Тренировочные данные
n_dim = features.shape[1]  # Вход нейросети
n1 = 100  # Количество нейронов скрытого слоя
# n2 = 50  # Количество нейронов скрытого слоя
n_outputs = 1  # Выходной нейрон

features_norm = normalize(features)  # Нормализация признаков
train_x = np.transpose(features_norm)
print(train_x.shape)
train_y = labels.reshape(1, len(labels))
X = tf.compat.v1.placeholder(tf.float32, [n_dim, None])  # Содержит матрицу X[xn, m] m - не объявляется
Y = tf.compat.v1.placeholder(tf.float32, [1, None])  # Содержит выходные значения размерности [1, m]
learning_rate = tf.compat.v1.placeholder(tf.float32, shape=())  # Содержит темп заучивания

hidden1 = create_layer(X, n1, activation=tf.sigmoid)
# hidden2 = create_layer(hidden1, n2, activation=tf.nn.relu)
# hidden3 = create_layer(hiddenl2, n3, activation=tf.sigmoid)
y_ = tf.sigmoid(hidden1)  # Вычисляет выходной нейрон
init = tf.compat.v1.global_variables_initializer()  # Инициализация переменных

cost = - tf.reduce_mean(Y * tf.math.log(y_) + (1 - Y) * tf.math.log(1 - y_))  # Определяет стоимостную функцию
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate).minimize(cost)
minibatch_size = 20
training_epochs = 10000
sees, cost_history = model(minibatch_size, training_epochs, train_x, train_y, logging_step=100, learning_r=0.001)

correct_predictionl = tf.equal(tf.greater(y_, 0.5), tf.equal(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_predictionl, tf.float32))
print(sees.run(accuracy, feed_dict={X: train_x, Y: train_y, learning_rate: 0.002}))
# print(sees.run(W))
sees.close()

plt.plot(cost_history)
plt.grid()
plt.show()
