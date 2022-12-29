from tensorflow import keras
from tensorflow.keras.layers import Dense

import numpy as np
import matplotlib.pyplot as plt


def far(value):
    return (9 / 5) * value + 32


c = np.array([-40, -10, 0, 8, 15, 22, 38])
f = np.array([-40, 14, 32, 46, 59, 72, 100])
model = keras.Sequential()
model.add(Dense(units=1, input_shape=(1,), activation='linear'))

model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(0.1))

history = model.fit(c, f, epochs=500, verbose=0)

plt.plot(history.history['loss'])
plt.grid(True)
plt.show()
print('Обучение завершено')

print(model.predict([c]))
print(model.get_weights())

c = int(input())
print(far(c))
