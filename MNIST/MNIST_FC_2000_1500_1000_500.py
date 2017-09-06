import time
import sys
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import tensorflow as tf


neu1 = 2000
neu2 = 1500
neu3 = 1000
neu4 = 500

x = tf.placeholder(tf.float32, [None, 784])
W1 = tf.Variable(tf.random_uniform([784, neu1], -1./784, 1./784))
b1 = tf.Variable(tf.random_uniform([neu1], -1.784, 1./784))
z1 = tf.nn.relu(tf.matmul(x, W1)+b1)

W2 = tf.Variable(tf.random_uniform([neu1, neu2], -1./neu1, 1./neu1))
b2 = tf.Variable(tf.random_uniform([neu2], -1./neu1, 1./neu1))
z2 = tf.nn.relu(tf.matmul(z1, W2)+b2)

W3 = tf.Variable(tf.random_uniform([neu2, neu3], -1./neu2, 1./neu2))
b3 = tf.Variable(tf.random_uniform([neu3], -1./neu2, 1./neu2))
z3 = tf.nn.relu(tf.matmul(z2, W3)+b3)

W4 = tf.Variable(tf.random_uniform([neu3, neu4], -1./neu3, 1./neu3))
b4 = tf.Variable(tf.random_uniform([neu4], -1./neu3, 1./neu3))
z4 = tf.nn.relu(tf.matmul(z3, W4)+b4)

W5 = tf.Variable(tf.random_uniform([neu4, 10], -1./neu4, 1./neu4))
b5 = tf.Variable(tf.random_uniform([10], -1./neu4, 1./neu4))
#z5 = tf.nn.relu(tf.matmul(z4, W5)+b5)

#W3 = tf.Variable(tf.random_uniform([neu, 10], -1./neu, 1./neu))
#b3 = tf.Variable(tf.random_uniform([10], -1./neu, 1./neu))

y = tf.nn.softmax(tf.matmul(z4,W5) + b5)

y_ = tf.placeholder(tf.float32, [None, 10])

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

#train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

init = tf.initialize_all_variables()

start = time.time()
sess = tf.Session()
sess.run(init)
for i in range(1200):
	batch_xs, batch_ys = mnist.train.next_batch(50)
	#print "episode ", i, "| cross entropy error: ", sess.run(cross_entropy, feed_dict={x: batch_xs, y_: batch_ys})
	sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
stop = time.time()
print "time is: ", stop-start

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))


