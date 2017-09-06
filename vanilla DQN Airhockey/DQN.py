import numpy as np
import tensorflow as tf
from utils.ReplayBuffer import ReplayBuffer
from utils.MDP_state import MDP_state
from collections import deque
import gym
import math
from utils.Networks import *
from utils.action_discretizer import *
import sys

# global definitions
OUT_DIR = "hockey/"
T = -0.5
Q_NET_SIZES = (40,100,100)
STATE_SIZE = 8
FRAMES = 1
MDP_STATE_SIZE = FRAMES* STATE_SIZE
ACTION_SIZE = 5		# in each axis
ACTION_DIM = 2
DISCRETIZATION = ACTION_SIZE ** ACTION_DIM
LEARNING_RATE = 0.001
GAMMA = 0.999
ANNEALING = 10000		# (5 episodes)
EPSILON = 0.1
BUFFER_SIZE = 1000000
MINI_BATCH = 32
BATCHES = 1
OBSERVATION_PHASE = 5		# (5 episdoes)
ENVIRONMENT = 'Hockey-v2'
EPISODES = 20000
STEPS = 200
SAVE_RATE = 200
C_STEPS = 200			# (30 episodes)
BUFFER_FILE = 'Replay_buffer'
DISPLAY = False

##################
# counter
##################
log_counter = Counter("log")
save_counter = Counter("save")

##################
# episode rewards
#################
reward = tf.placeholder(tf.float32, [1], name="reward")
save_reward = reward[0]
reward_sum = tf.scalar_summary('Reward',save_reward)

##################
# Q networks
##################
Q = DeterministicMLP("Q", MDP_STATE_SIZE, Q_NET_SIZES, DISCRETIZATION, LEARNING_RATE, GAMMA, True)
Q_target = DeterministicMLP("Q_target", MDP_STATE_SIZE, Q_NET_SIZES, DISCRETIZATION, LEARNING_RATE, GAMMA, True)


saver = tf.train.Saver()
init = tf.initialize_all_variables()
summary = tf.merge_all_summaries()

sess = tf.Session()

logger = tf.train.SummaryWriter(OUT_DIR, sess.graph)

# initialize variables (and target network)
sess.run(init)
Ws,bs = Q.get_weights()
Q_target.assign(sess, Ws,bs)

ann_fric = (1-EPSILON)/ANNEALING
EXP_PROB = 1

# initialize environment
env = gym.make(ENVIRONMENT)

# initialize mdp state structure
mdp = MDP_state(STATE_SIZE, FRAMES)

# initialize replay buffer
R = ReplayBuffer(MDP_STATE_SIZE, 1, BUFFER_SIZE)
buf = R.LoadBuffer(OUT_DIR+BUFFER_FILE)
if buf:
	EXP_PROB = EPSILON
	populated = R.GetOccupency()
	print("Replay buffer loaded from disk, occupied: " + str(populated))
else:
	print("Creating new replay buffer")

# load saved model
ckpt = tf.train.get_checkpoint_state(OUT_DIR)
if ckpt and ckpt.model_checkpoint_path:
	saver.restore(sess,ckpt.model_checkpoint_path)
	print("Model loaded from disk")

# define action discretization
max_a = env.action_space.high[0]
min_a = env.action_space.low[0]

act = actions(ACTION_SIZE, max_a)
actions_deque,_ = act.get_action()
discretizer = Discretizer(actions_deque)

# main learning loop
print("Starting to learn in environment: " + ENVIRONMENT)
steps = 0
for episode_i in xrange(1,EPISODES+1):
	st = env.reset()
	mdp.add_frame(st)
	st = mdp.get_MDP_state()
	totalR = 0
	for t in xrange(1,STEPS+1):
		if DISPLAY:
			env.render()

		# select action
		exp = np.random.uniform()
		if exp > EXP_PROB:
			q_vector = Q.evaluate(sess, st)
			a_index = np.argmax(q_vector)
		else:
			a_index = np.random.randint(0,DISCRETIZATION)
		at = actions_deque[a_index]
		if EXP_PROB > EPSILON:
			EXP_PROB -= ann_fric

		# execute action
		st_next, rt, Done, _ = env.step(at)
		mdp.add_frame(st_next)
		rt = rt+T
		st_next = mdp.get_MDP_state()
		if Done:
			dt = 1
		else:
			dt = 0
		totalR += rt
		
		# store transition
		R.StoreTransition(st, np.array([a_index]), np.array([rt]), st_next, dt)
		st = st_next

		if episode_i > OBSERVATION_PHASE:
			for mini_batch in xrange(BATCHES):
				# sample mini batch
				s_batch, a_batch, r_batch, stag_batch, terminal_batch = R.SampleMiniBatch(MINI_BATCH)
				
				Y = Q.evaluate(sess, s_batch)				
	
				Q_next = Q_target.evaluate(sess, stag_batch)
				Q_next_max = np.amax(Q_next,1)

				a_batch = a_batch.astype(int)
				for i in range(MINI_BATCH):
					Y[i,a_batch[i,0]] = r_batch[i,0] + GAMMA*Q_next_max[i] * (1-terminal_batch[i])
			
				# train on estimated Q next and rewards
				error = Q.train(sess, s_batch, Y)

		if Done is True:
			break
		

	if steps >= C_STEPS:
		s,bs = Q.get_weights()
		Q_target.assign(sess, Ws,bs)
		print ('updating traget network')
		steps = 0
	steps += 1
		
	r = np.array([totalR])
	log = sess.run(summary,{reward:r})
	log_counter.increment(sess)
	logger.add_summary(log,log_counter.evaluate(sess))
	print ("episode %d/%d (%d), reward: %f" % (episode_i, EPISODES, log_counter.evaluate(sess), totalR))

	if episode_i % SAVE_RATE == 0:
		save_counter.increment(sess)
		saver.save(sess,OUT_DIR+"model.ckpt", global_step=save_counter.evaluate(sess))
		R.SaveBuffer(OUT_DIR+BUFFER_FILE)
		print "model saved, replay buffer: ", R.GetOccupency()		

Ws,bs = Q.get_weights()
Q_target.assign(sess, Ws,bs)
save_counter.increment(sess)
saver.save(sess,OUT_DIR+"model.ckpt", global_step=save_counter.evaluate(sess))
R.SaveBuffer(OUT_DIR+BUFFER_FILE)

sess.close()

