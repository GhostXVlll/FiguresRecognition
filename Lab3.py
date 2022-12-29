from tensorflow import keras
from keras.layers import Dense, Flatten
from keras.datasets import mnist
import imageio.v2 as imageio

import numpy as np
import matplotlib.pyplot as plt


def learning():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train / 255
    x_stest = x_test / 255

    y_train_cat = keras.utils.to_categorical(y_train, 10)
    y_test_cat = keras.utils.to_categorical(y_test, 10)

    """plt.figure(figsize=(10, 5))  # Вывод изображений из обучающей выборки
    for i in range(25):
        plt.subplot(5, 5, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(x_train[i], cmap=plt.cm.binary)
    plt.show()"""

    model = keras.Sequential([
        Flatten(input_shape=(28, 28, 1)),
        Dense(128, activation='relu'),  # Скрытый слой с функцией активации relu
        Dense(10, activation='softmax')  # Выходной слой с функцией активации softmax
    ])

    print(model.summary())  # Вывод структуры сети в консоль

    # Компилляция нейронки с оптимизацией по Adam и критерием - категориальная кросс-энтропия
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train, y_train_cat, batch_size=32, epochs=5, validation_split=0.2)  # Обучение нейросети
    # batch_size - количество картинок между корректировками весовых коэффициентов (размер батча)
    # validation_split разбиение выборки на обучающую и проверочную (80% - обуч, 20% - тест)
    # epochs количество эпох обучения нейросети

    model.evaluate(x_test, y_test_cat)
    model.save('model/')  # Сохранение модели

    # Проверка распознавания цифр
    n = 0
    x = np.expand_dims(x_test[n], axis=0)
    res = model.predict(x)
    print(res)
    print(f"Распознанная цифра: {np.argmax(res)}")
    plt.imshow(x_test[n], cmap=plt.cm.binary)
    plt.show()
    # Распознавание всей тестовой выборки
    pred = model.predict(x_test)
    pred = np.argmax(pred, axis=1)
    print(pred.shape)
    print(pred[:20])
    print(y_test[:20])
    # Выделение ошибок
    mask = pred == y_test
    x_false = x_test[~mask]
    p_false = pred[~mask]
    print(x_false.shape)
    """# Вывод первых 5 неверных изображений
    for idx in range(5):
        print("Значение сети: " + str(p_false[idx]))
        plt.imshow(x_false[idx], cmap=plt.cm.binary)
        plt.show()"""


def prediction(model, filename, display=True):  # Распознание цифры из файла
    image = imageio.imread("image.png")
    image = np.mean(image, 2, dtype=float)
    image = image / 255
    if display:
        plt.xticks([])
        plt.yticks([])
        plt.imshow(image, cmap=plt.cm.binary)
        plt.xlabel(filename)
        plt.show()

    image = np.expand_dims(image, 0)
    return np.argmax(model.predict(image))


def main():
    try:
        model = keras.models.load_model('model/')
        print(f'Model {model} was loaded')

        filename = 'image.png'
        print('filaneme: ', filename, '\tPrediction: ', prediction(model, filename, False))

    except Exception:
        print("Model not found")
        learning()


if __name__ == "main":
    main()
