import tensorflow as tf
import numpy as np

model = tf.keras.models.Sequential(tf.keras.layers.Dense(units=1, input_shape=[1]))
model.compile(optimizer='sgd', loss='mean_squared_error')

x = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=float)
y = np.array([50000.0, 100000.0, 150000.0, 200000.0, 250000.0, 300000.0, 350000.0], dtype=float)

model.fit(x, y, epochs=1952)
print(model.predict([7.0]))