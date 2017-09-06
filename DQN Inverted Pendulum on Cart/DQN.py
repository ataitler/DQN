import numpy as np
import tensorflow as tf
from ReplayBuffer import ReplayBuffer
import gym
import math

# global definitions
OUT_DIR = "results_pen_test/"
Q_NET_SIZES = (24,24)
STATE_SIZE = 3
ACTION_SIZE = 1
DISCRETIZATION = 11
LEARNING_RATE = 0.001
GAMMA = 0.999
ANNEALING = 1000		# (5 episodes)
EPSILON = 0.1
BUFFER_SIZE = 100000
MINI_BATCH = 16
BATCHES = 64
OBSERVATION_PHASE = 600		# (3 episdoes)
ENVIRONMENT = 'Pendulum-v0'
EPISODES = 500
STEPS = 200
SAVE_RATE = 30
C_STEPS = 30			# (30 episodes)
BUFFER_FILE = 'Replay_buffer'
DISPLAY = True

##################
# counter
##################
save_counter = tf.Variable(0, name="save_counter")
log_counter = tf.Variable(0, name = "log_counter")
one = tf.constant(1, name="one")
IncrementSaveCounter = tf.assign_add(save_counter,one)
IncrementLogCounter = tf.assign_add(log_counter,one)

##################
# episode rewards
#################
reward = tf.placeholder(tf.float32, [1], name="reward")
save_reward = reward[0]
reward_sum = tf.scalar_summary('Reward',save_reward)

hot_ind = tf.placeholder(tf.int64, [None])
hot1 = tf.one_hot(hot_ind, DISCRETIZATION, on_value=1, off_value=0)

##################
# Q networks
##################
state_input = tf.placeholder(tf.float32, [None, STATE_SIZE], name="Q_state_input")

prevDim = STATE_SIZE
prevOut = state_input

# Q network
W1 = tf.Variable(tf.random_uniform([STATE_SIZE,Q_NET_SIZES[0]], -1/math.sqrt(STATE_SIZE), 1/math.sqrt(STATE_SIZE)), name="W1")
W1_sum = tf.histogram_summary("Q/W1",W1)
W2 = tf.Variable(tf.random_uniform([Q_NET_SIZES[0],Q_NET_SIZES[1]], -1/math.sqrt(Q_NET_SIZES[0]), 1/math.sqrt(Q_NET_SIZES[0])), name="W2")
W2_sum = tf.histogram_summary("Q/W2",W2)
W3 = tf.Variable(tf.random_uniform([Q_NET_SIZES[1], DISCRETIZATION], -1/math.sqrt(Q_NET_SIZES[1]), 1/math.sqrt(Q_NET_SIZES[1])), name="W3")
W3_sum = tf.histogram_summary("Q/W3",W3)

b1 = tf.Variable(tf.random_uniform([Q_NET_SIZES[0]], -1/math.sqrt(STATE_SIZE), 1/math.sqrt(STATE_SIZE)), name="b1")
b1_sum = tf.histogram_summary("Q/b1",b1)
b2 = tf.Variable(tf.random_uniform([Q_NET_SIZES[1]], -1/math.sqrt(Q_NET_SIZES[0]), 1/math.sqrt(Q_NET_SIZES[0])), name="b2")
b2_sum = tf.histogram_summary("Q/b2",b2)
b3 = tf.Variable(tf.random_uniform([DISCRETIZATION], -1/math.sqrt(Q_NET_SIZES[1]), 1/math.sqrt(Q_NET_SIZES[1])), name="b3")
b3_sum = tf.histogram_summary("Q/b3",b3)

# target Q network
W1_target = tf.Variable(tf.random_uniform([STATE_SIZE,Q_NET_SIZES[0]], -1/math.sqrt(STATE_SIZE), 1/math.sqrt(STATE_SIZE)), name="W1_target")
W1_target_sum = tf.histogram_summary("Q_target/W1",W1_target)
W2_target = tf.Variable(tf.random_uniform([Q_NET_SIZES[0],Q_NET_SIZES[1]], -1/math.sqrt(Q_NET_SIZES[0]), 1/math.sqrt(Q_NET_SIZES[0])), name="W2_target")
W2_target_sum = tf.histogram_summary("Q_target/W2",W2_target)
W3_target = tf.Variable(tf.random_uniform([Q_NET_SIZES[1], DISCRETIZATION], -1/math.sqrt(Q_NET_SIZES[1]), 1/math.sqrt(Q_NET_SIZES[1])), name="W3_target")
W3_target_sum = tf.histogram_summary("Q_target/W3",W3_target)

b1_target = tf.Variable(tf.random_uniform([Q_NET_SIZES[0]], -1/math.sqrt(STATE_SIZE), 1/math.sqrt(STATE_SIZE)), name="b1_target")
b1_target_sum = tf.histogram_summary("Q_target/b1",b1_target)
b2_target = tf.Variable(tf.random_uniform([Q_NET_SIZES[1]], -1/math.sqrt(Q_NET_SIZES[0]), 1/math.sqrt(Q_NET_SIZES[0])), name="b2_target")
b2_target_sum = tf.histogram_summary("Q_target/b2",b2_target)
b3_target = tf.Variable(tf.random_uniform([DISCRETIZATION], -1/math.sqrt(Q_NET_SIZES[1]), 1/math.sqrt(Q_NET_SIZES[1])), name="b3_target")
b3_target_sum = tf.histogram_summary("Q_target/b3",b3_target)

z1 = tf.nn.relu(tf.matmul(state_input, W1) + b1, name="z1")
z2 = tf.nn.relu(tf.matmul(z1, W2) + b2, name="z2")
Q = tf.add(tf.matmul(z2,W3), b3, name="Q")

z1_target = tf.nn.relu(tf.matmul(state_input, W1_target) + b1_target, name="z1_target")
z2_target = tf.nn.relu(tf.matmul(z1_target, W2_target) + b2_target, name="z2_target")
Q_target = tf.add(tf.matmul(z2_target, W3_target), b3_target, name="Q_target")

# training procedure
y_estimate = tf.placeholder(tf.float32, [None, DISCRETIZATION], name = "y_estimate")

Q_cost = tf.reduce_mean(tf.reduce_sum(tf.square(y_estimate - Q),1))

#optimization of Q
#Q_train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(Q_cost)
Q_train_step = tf.train.RMSPropOptimizer(LEARNING_RATE, GAMMA,0.0,1e-6).minimize(Q_cost)

saver = tf.train.Saver()
init = tf.initialize_all_variables()
summary = tf.merge_all_summaries()

sess = tf.Session()

logger = tf.train.SummaryWriter(OUT_DIR, sess.graph)

# initialize variables (and target network)
sess.run(init)
sess.run([W1_target.assign(W1),W2_target.assign(W2),W3_target.assign(W3),
          b1_target.assign(b1),b2_target.assign(b2),b3_target.assign(b3)])

# initialize environment
env = gym.make(ENVIRONMENT)

# initialize replay buffer
R = ReplayBuffer(STATE_SIZE, ACTION_SIZE, BUFFER_SIZE)
buf = R.LoadBuffer(OUT_DIR+BUFFER_FILE)
if buf:
	populated = R.GetOccupency()
	print("Replay buffer loaded from disk, occupied: " + str(populated))

# load saved model
ckpt = tf.train.get_checkpoint_state(OUT_DIR)
if ckpt and ckpt.model_checkpoint_path:
	saver.restore(sess,ckpt.model_checkpoint_path)
	print("Model loaded from disk")

# define action discretization
a_max = env.action_space.high
a_min = env.action_space.low
actions = np.linspace(a_min, a_max, DISCRETIZATION)

ann_fric = (1-EPSILON)/ANNEALING
EXP_PROB = 1

# main learning loop
print("Starting to learn in environment: " + ENVIRONMENT)
steps = 0
for episode_i in xrange(1,EPISODES+1):
	st = env.reset()
	totalR = 0
	for t in xrange(1,STEPS+1):
		if DISPLAY:
			env.render()

		# select action
		exp = np.random.uniform()
		if exp > EXP_PROB:
			q_vector = sess.run(Q, feed_dict={state_input:st.reshape(1,STATE_SIZE)})
			a_index = np.argmax(q_vector)
			at = actions[a_index]
		else:
			a_index = np.random.randint(0,DISCRETIZATION)
			at = actions[a_index]
		if EXP_PROB > EPSILON:
			EXP_PROB -= ann_fric

		# execute action
		st_next, rt, Done, _ = env.step(np.array([at]))
		if Done:
			dt = 1
		else:
			dt = 0
		totalR += rt
		
		# store transition
		R.StoreTransition(st, np.array([a_index]), rt, st_next, dt)
		st = st_next
		
		if episode_i <= 5:
			continue

	for mini_batch in xrange(BATCHES):
		# sample mini batch
		s_batch, a_batch, r_batch, stag_batch, terminal_batch = R.SampleMiniBatch(MINI_BATCH)
		Y = sess.run(Q, {state_input:s_batch})

		Q_next = sess.run(Q_target, feed_dict={state_input:stag_batch})
		Q_next_max = np.amax(Q_next,1)

		a_batch = a_batch.astype(int)
		for i in range(MINI_BATCH):
			Y[i,a_batch[i,0]] = r_batch[i,0] + GAMMA*Q_next_max[i] * (1-terminal_batch[i])
			
		# train on estimated Q next and rewards
		sess.run(Q_train_step, {state_input:s_batch, y_estimate:Y})

	if steps >= C_STEPS:
		sess.run([W1_target.assign(W1),W2_target.assign(W2),W3_target.assign(W3),
		          b1_target.assign(b1),b2_target.assign(b2),b3_target.assign(b3)])
		print ('updating traget network')
		steps = 0
	steps += 1
		
	r = np.array([totalR])
	log = sess.run(summary,{reward:r})
	sess.run(IncrementLogCounter)
	logger.add_summary(log,log_counter.eval(sess))
	print ("episode %d/%d (%d), reward: %f" % (episode_i, EPISODES, log_counter.eval(sess), totalR))

	if episode_i % SAVE_RATE == 0:
		sess.run(IncrementSaveCounter)
		saver.save(sess,OUT_DIR+"model.ckpt", global_step=save_counter.eval(sess))
		R.SaveBuffer(OUT_DIR+BUFFER_FILE)
		



