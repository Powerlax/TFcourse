import tensorflow as tf
import numpy as np

model = tf.keras.models.Sequential([tf.keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd', loss='mean_squared_error')

x = np.array([-1.0, 0.0, 1.0, 2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0],dtype=float)
y = np.array([-3.0, -1.0,1.0,3.0,5.0,7.0,9.0,11.0,13.0,15.0,17.0,19.0],dtype=float)

model.fit(x,y,epochs=2000)

print(model.predict([11.0]))