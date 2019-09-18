import numpy as np
import tensorflow as tf

mask = tf.cast(tf.convert_to_tensor(np.array([[1, 1, 0], [1, 0, 1]])), dtype=tf.int32)

