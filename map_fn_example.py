import tensorflow as tf
import numpy as np

def alternate(x):
    # a = dict(zip([1, 2, 3], [2, 4, 6]))
    return x[0] * x[1]


elems = (np.array([1, 2, 3]), np.array([-1, 1, -1]))
# alternate = tf.map_fn(lambda x: x[0] * x[1], elems, dtype=tf.int64)

alter = tf.map_fn(alternate, elems=elems, dtype=tf.int64)

sess = tf.Session()

print sess.run(alter)
# print a