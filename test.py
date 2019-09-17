import numpy as np
import tensorflow as tf

c = np.zeros([5, 10, 128])
c = tf.cast(tf.convert_to_tensor(c), dtype=tf.float32)

pos = tf.get_variable('pos', [10, 128], dtype=tf.float32)

c += pos

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)
print(sess.run(p))
print(sess.run(c))