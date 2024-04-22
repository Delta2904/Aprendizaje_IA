import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

l0 = Dense(units=1, input_shape=[1])

model = Sequential(l0)
model.compile(optimizer='sgd', loss='mean_squared_error')

xs = np.array([-5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=float)
ys = np.array([-11.0, -9.0, -7.0, -5.0, -3.0, -1.0, 1.0, 3.0, 5.0, 7.0, 9.0, 11.0], dtype=float)

model.fit(xs, ys, epochs=5000)

print(model.predict([10.0]))
print('Lo que aprendió el modelo fue: {}'.format(l0.get_weights()))
