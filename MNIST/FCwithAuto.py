import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])

#FC autoencoder

#W_auto1 = tf.Variable(tf.random_uniform([784, 784], -1/784, 1/784))
#W_auto2 = tf.Variable(tf.random_uniform([784, 784], -1/784, 1/784))
#W_auto3 = tf.Variable(tf.random_uniform([784, 784], -1/784, 1/784))
#b_auto1 = tf.Variable(tf.random_uniform([784], -1/784, 1/784))
#b_auto2 = tf.Variable(tf.random_uniform([784], -1/784, 1/784))
#b_auto3 = tf.Variable(tf.random_uniform([784], -1/784, 1/784))
W_auto1 = tf.Variable(tf.truncated_normal([784, 784], stddev=0.01))
b_auto1 = tf.Variable(tf.truncated_normal([784], stddev=0.01))
W_auto2 = tf.Variable(tf.truncated_normal([784, 784], stddev=0.01))
b_auto2 = tf.Variable(tf.truncated_normal([784], stddev=0.01))
W_auto3 = tf.Variable(tf.truncated_normal([784, 784], stddev=0.01))
b_auto3 = tf.Variable(tf.truncated_normal([784], stddev=0.01))

x_auto1 = tf.nn.relu(tf.matmul(x, W_auto1) + b_auto1)
x_auto2 = tf.nn.relu(tf.matmul(x_auto1, W_auto2) + b_auto2)
x_auto3 = tf.add(tf.matmul(x_auto2, W_auto3), b_auto3)

#loss_auto = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=x, logits=x_auto3))
loss_auto = tf.reduce_mean(tf.reduce_sum(tf.square(x_auto3-x),1))
train_auto = tf.train.RMSPropOptimizer(0.001).minimize(loss_auto)

#FC classifier

#W = tf.Variable(tf.truncated_normal([784, 10], stddev=0.01))
#b = tf.Variable(tf.truncated_normal([10], stddev=0.01))
#y = tf.nn.softmax(tf.matmul(x_auto2, W) + b)
#y = tf.reduce_mean(tf.matmul(x_auto2,W)+b)

#loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
#train = tf.train.RMSPropOptimizer(0.001).minimize(loss, var_list=[W, b])

#training

init = tf.initialize_all_variables()
#init = tf.global_variables_initializer()

with tf.Session() as sess:
  sess.run(init)
#train autoencoder
  print "training autoencoder part"
  for i in range(10000):
    batch_xs, batch_ys = mnist.train.next_batch(10)
    sess.run(train_auto, feed_dict={x: batch_xs, y_: batch_ys})
    print "Step ", i, " Generation error:", sess.run(loss_auto,{x:batch_xs, y_:batch_ys})

#train classifier
#  print "training classifier"
#  for i in range(1000):
#    batch_xs, batch_ys = mnist.train.next_batch(100)
#    sess.run(train, feed_dict={x: batch_xs, y_: batch_ys})
#    print "Step ", i, " Classification error:", sess.run(loss,{x:batch_xs, y_:batch_ys})

#evaluating

  #correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  #accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
 # print(sess.run(loss, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
  #print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
