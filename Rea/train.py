import numpy as np
import tensorflow as tf
from Networks import *
import matplotlib.pyplot as plt


OUT_DIR = "results"
INPUT_SIZE = 14
OUTPUT_SIZE = 1
LAYERS = (32,32)
LEARNING_RATE = 0.0001
GAMMA = 0.99
EPOCHS = 30
MINI_BATCH = 32
TEST_SIZE = 3000

a = np.load('npdata.npz')
X = a['X']
Y = a['Y']
Y = Y.astype(np.float)
MAX = Y.shape[0]

test_ind = np.random.randint(MAX, size = TEST_SIZE)
test_x = X[test_ind] 
test_y = Y[test_ind]
err = []

net = DeterministicMLP("reg", INPUT_SIZE, LAYERS, OUTPUT_SIZE, LEARNING_RATE, GAMMA, False, 0.01, 0.0)

saver = tf.train.Saver()
init = tf.initialize_all_variables()
summary = tf.merge_all_summaries()

sess = tf.Session()
logger = tf.train.SummaryWriter(OUT_DIR, sess.graph)

sess.run(init)

error = np.linalg.norm(net.evaluate(sess,test_x) - test_y)/TEST_SIZE
err.append(error)
print 'starting error: ', error
for epoch in xrange(EPOCHS):
	mb = 0
	while mb < MAX:
		if mb + MINI_BATCH < MAX:
			images = X[mb:mb+MINI_BATCH]
			labels = Y[mb:mb+MINI_BATCH]
		else:
			images = X[mb:MAX-1]
			labels = Y[mb:MAX-1]
		mb = mb+MINI_BATCH
		net.train(sess, images, labels)
	error = np.linalg.norm(net.evaluate(sess,test_x) - test_y)/TEST_SIZE
	err.append(error)
	print 'epoch ', epoch+1, ' error: ', error
	saver.save(sess,OUT_DIR+"model.ckpt", global_step=epoch)

#plt.plot(test_y, net.evaluate(sess,test_x))
plt.plot(err)
plt.show()





