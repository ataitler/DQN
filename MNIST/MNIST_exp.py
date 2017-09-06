import os
import numpy as np
import struct

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

import tensorflow as tf

########################################################

def print_conv_layer_to_file(strings_output_file, binary_output_file, input_weights_to_print, bias_to_print):
	row_len = input_weights_to_print.shape[0]
	print("row_len: ", row_len)
	col_len = input_weights_to_print.shape[1]
	print("col_len: ", col_len)
	inputs_num = input_weights_to_print.shape[2]
	print("input maps num: ", inputs_num)
	outputs_num = input_weights_to_print.shape[3]
	print("output maps num: ", outputs_num)

	for output_map_index in range(outputs_num):
		for input_map_index in range(inputs_num):
			for neuron_index in range(col_len):
				#print("neuron_index: ", neuron_index)
				for weight_index in range(row_len):
					#print("neuron_index: ", neuron_index, " weight_index: ", weight_index)
					strings_output_file.write(str(input_weights_to_print[weight_index][neuron_index][input_map_index][output_map_index]))
					strings_output_file.write('\n')
					binary_output_file.write(struct.pack('<f',input_weights_to_print[weight_index][neuron_index][input_map_index][output_map_index]))
				
		#print("str(W_fc2_to_print[1023][neuron_index])" ,str(input_weights_to_print[1023][neuron_index]))
		#print("str(bias_to_print[output_map_index]): ", str(bias_to_print[output_map_index]))
		strings_output_file.write(str(bias_to_print[output_map_index]))
		strings_output_file.write('\n')
		binary_output_file.write(struct.pack('<f', bias_to_print[output_map_index]))

###-------------------------------------------

def print_fully_connected_layer_to_file(strings_output_file, binary_output_file, input_weights_to_print, bias_to_print):
	row_len = input_weights_to_print.shape[0]
	print("row_len is: ", row_len)
	col_len = input_weights_to_print.shape[1]
	print("col_len is: ", col_len)

	for neuron_index in range(col_len):
		#print("neuron_index: ", neuron_index)
		for weight_index in range(row_len):
			#print("neuron_index: ", neuron_index, " weight_index: ", weight_index)
			strings_output_file.write(str(input_weights_to_print[weight_index][neuron_index]))
			strings_output_file.write('\n')
			binary_output_file.write(struct.pack('<f',input_weights_to_print[weight_index][neuron_index]))
		
		#print("str(W_fc2_to_print[1023][neuron_index])" ,str(input_weights_to_print[1023][neuron_index]))
		#print("str(bias_to_print[neuron_index]): ", str(bias_to_print[neuron_index]))
		strings_output_file.write(str(bias_to_print[neuron_index]))
		strings_output_file.write('\n')
		binary_output_file.write(struct.pack('<f', bias_to_print[neuron_index]))

###-------------------------------------------

def print_layer_weights_to_file(input_weights, bias, layer_name, sess, conv=False):
	input_weights_to_print =  input_weights.eval(sess)
	bias_to_print =  bias.eval(sess)
	#print(np.array2string(b_fc2_to_print, precision=5, separator=',', suppress_small=False))

	strings_output_file = open('Roman/'+layer_name+'_strings', 'w')
	binary_output_file = open('Roman/'+layer_name+'_binary', 'w')

	if not conv:
		print_fully_connected_layer_to_file(strings_output_file, binary_output_file, input_weights_to_print, bias_to_print)
	else:
		print_conv_layer_to_file(strings_output_file, binary_output_file, input_weights_to_print, bias_to_print)

	#output_file.write(np.array2string(W_fc2_to_print, precision=5, separator=',', suppress_small=False))
	strings_output_file.close()
	binary_output_file.close()

###-------------------------------------------

def print_test_set_labels_output(labels):
	#input_weights_to_print =  input_weights.eval(sess)
	#print(np.array2string(b_fc2_to_print, precision=5, separator=',', suppress_small=False))

	labels_output_file = open('Roman/test_set_labels_output', 'w')

	for i in range(len(labels)):
		labels_output_file.write(str(i))
		labels_output_file.write(', ')
		labels_output_file.write(str(labels[i]))
		labels_output_file.write('\n')

	#output_file.write(np.array2string(W_fc2_to_print, precision=5, separator=',', suppress_small=False))
	labels_output_file.close()
###-------------------------------------------

def print_conv_activations_to_file(output_feature_maps_to_print, layer_name, sess):
	print("Printing conv layer activations", layer_name)
	#output_feature_maps_to_print =  output_feature_maps.eval(sess)
	#print(np.array2string(b_fc2_to_print, precision=5, separator=',', suppress_small=False))

	strings_output_file = open('Roman/'+layer_name+'_output_feature_maps_strings', 'w')
	binary_output_file = open('Roman/'+layer_name+'_output_feature_maps_binary', 'w')

	row_len = output_feature_maps_to_print.shape[0]
	print("row_len: ", row_len)
	col_len = output_feature_maps_to_print.shape[1]
	print("col_len: ", col_len)
	output_feature_maps_num = output_feature_maps_to_print.shape[2]
	print("Output maps num: ", output_feature_maps_num)

	for output_map_index in range(output_feature_maps_num):
		for col_index in range(col_len):
			#print("neuron_index: ", neuron_index)
			for row_index in range(row_len):
				#print("row_index: ", row_index, " col_index: ", col_index)
				strings_output_file.write(str(output_feature_maps_to_print[row_index][col_index][output_map_index]))
				strings_output_file.write('\n')
				binary_output_file.write(struct.pack('<f',output_feature_maps_to_print[row_index][col_index][output_map_index]))
				

	strings_output_file.close()
	binary_output_file.close()
########################################################


def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape = [None, 10])
	
# reshape input
x_image = tf.reshape(x, [-1, 28, 28, 1])

# first convolutional later
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)
# second convolutional layer
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# fully connected layer
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# softmax
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

outputs = tf.argmax(y_conv,1)


cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver()

sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
for train_iter in range(10):

	for i in range(1200):
		batch = mnist.train.next_batch(50)
		if i%100 == 0:
			train_accuracy = sess.run(accuracy, feed_dict={x: batch[0], y_:batch[1], keep_prob:0.1})
			print("step %d, training accuracy %f"%(i,train_accuracy))
		sess.run(train_step, feed_dict={x:batch[0], y_:batch[1], keep_prob:0.5})

test_accuracy = sess.run(accuracy, feed_dict={x:mnist.test.images, y_:mnist.test.labels, keep_prob:1.0})
print("test accuracy %g"%test_accuracy)
labels = [0, 4, 6]
out = sess.run(outputs, {x:mnist.test.images, y_:mnist.test.labels,keep_prob:1.0})
print_test_set_labels_output(out)

saver.save(sess, "Roman/model.ckpt")

np.set_printoptions(threshold='nan')
print_layer_weights_to_file(W_conv1, b_conv1, "conv1", sess, conv=True)
print_layer_weights_to_file(W_conv2, b_conv2, "conv2", sess, conv=True)
print_layer_weights_to_file(W_fc1, b_fc1, "FC1", sess)
print_layer_weights_to_file(W_fc2, b_fc2, "Output", sess)

print mnist.test.images.shape
conv1_output_feature_maps = sess.run(h_conv1, {x:mnist.test.images})[0]
print "finished calculating conv1 activations, conv1_output_feauture_maps.shape: ", conv1_output_feature_maps.shape
print_conv_activations_to_file(conv1_output_feature_maps, "conv1", sess)

conv2_output_feature_maps = sess.run(h_conv2, {x:mnist.test.images})[0]
print "finished calculating conv2 activations, conv2_output_feauture_maps.shape: ", conv2_output_feature_maps.shape
print_conv_activations_to_file(conv2_output_feature_maps, "conv2", sess)
