import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio


def prediction(model, filename, display=True):
    image = imageio.imread(filename)
    image = np.mean(image, 2, dtype=float)
    image = image / 255
    """if display:
        plt.xticks([])
        plt.yticks([])
        plt.imshow(image, cmap=plt.cm.binary)
        plt.xlabel(filename)
        plt.show()"""

    image = np.expand_dims(image, 0)
    return np.argmax(model.predict(image))


mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

""" print(type(x_test)) # Вид массива

print(x_test.shape) # Размар массива
print(y_test.shape) # Размер массива
"""
x_train = x_train / 255  # Нормальизация x_train и y_train в размерность (0...1)
y_train = y_train / 255

plt.figure(figsize=(8, 8)) # Вывод 16 первых цифр из обучающей выборки
for i in range(16):
    plt.subplot(4, 4, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_train[i], cmap=plt.cm.binary)
    plt.colorbar()
    plt.xlabel(int(y_train[i]*255))
# plt.show()

# ################## Create model
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(x_train, y_train, epochs=5)

print(model.evaluate(x_test, y_test))

# загрузка изображения
filename = 'image.png'
print('filaneme: ', filename, '\tPrediction: ', prediction(model, filename, False))
