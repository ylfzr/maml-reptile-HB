import tensorflow as tf

a = tf.Variable([0], name='hahaha')
print a.name[:-2]
# print str(a.name)[1:-2]