import tensorflow as tf
import numpy as np
from tensorbayesfeul.layers import linear
from tensorbayesfeul.elements import placeholder, constant

x = placeholder([3, 20, 5])
y = placeholder([3, 20,], dtype=tf.int32)
y_onehot = tf.one_hot(y, 2)
lr = 0.01
# global train_step_task
def mlp_task(inp):

    input, label = inp
    # print input, label
    h = linear(input, 4)
    out_put = linear(h, 2)
    accuracy = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(out_put), 1), tf.argmax(label, 1))
    loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=out_put, labels=label))

    t_vars = tf.trainable_variables()
    # num_var = len(t_vars)
    weights = dict([(t_var.name, t_var) for t_var in t_vars])
    # print weights.keys()

    grads = tf.gradients(loss, list(weights.values()))
    gradients = dict(zip(weights.keys(), grads))
    gradient_values = [gradients[key] for key in gradients.keys()]
    # print gradients.keys()
    # print gradients.values()
    # fast_weights = dict(zip(weights.keys(), [weights[key] - lr * gradients[key] for key in weights.keys()]))

    optimizer = tf.train.AdamOptimizer(lr)
    train_step_task = optimizer.apply_gradients(zip(gradients.values(), weights.values()))

    return loss, accuracy, tuple(gradient_values)



# Data prepare

# x_data = 0.1 * np.random.normal(size=[20, 5])
# x_data[10:,:] = x_data[10:,:] + 1
# y_data = np.zeros([20,])
# y_data[10:] = y_data[10:] + 1
x_data = 0.1 * np.random.normal(size=[3,20, 5])
x_data[:,10:,:] = x_data[:,10:,:] + 1
y_data = np.zeros([3,20,])
y_data[:,10:] = y_data[:,10:] + 1

num_step = 1

# loss, accuracy, train_step = mlp_task((x, y_onehot))


data_type = tuple([tf.float32, tf.float32, tuple([tf.float32] * 4)]) # get the gradients in this way
results = tf.map_fn(mlp_task, elems=(x, y_onehot), dtype=data_type, parallel_iterations=3)


sess = tf.Session()
sess.run(tf.global_variables_initializer())


for i in range(num_step):
    # loss_np, accuracy_np, _ = sess.run([loss, accuracy, train_step], feed_dict={x:x_data, y:y_data})
    losses_np, accuracies_np, gradients = sess.run(results, feed_dict={x: x_data, y: y_data})
    gradients_list = list(gradients)
    print gradients_list[0].shape
    print gradients_list[1].shape
    print gradients_list[2].shape
    print gradients_list[3].shape


# a = ('a', 'b', 'c')
# b = [a, 'd', 'e']
# c = (a, 'd', 'e')
# print b
# print c





