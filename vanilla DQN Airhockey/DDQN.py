import numpy as np
import tensorflow as tf
from utils.ReplayBuffer import ReplayBuffer
from utils.Logger import Logger
from utils.MDP_state import MDP_state
from collections import deque
import gym
import math
from utils.Networks import *
from utils.action_discretizer import *
import sys
from utils.Simulator import *
import matplotlib.pyplot as plt


# global definitions
OUT_DIR = "hockey_DQN_200_V4/"
T = -0.5
Q_NET_SIZES = (40,100,100)
STATE_SIZE = 8
FRAMES = 1
MDP_STATE_SIZE = FRAMES* STATE_SIZE
ACTION_SIZE = 5		# in each axis
ACTION_DIM = 2
DISCRETIZATION = ACTION_SIZE ** ACTION_DIM
LEARNING_RATE = 0.00025
GAMMA = 0.999
ANNEALING = 10000		# (5 episodes)
EPSILON = 0.1
BUFFER_SIZE = 200000
MINI_BATCH = 64
BATCHES = 1
V_EST = 500
OBSERVATION_PHASE = 5		# (5 episdoes)
ENVIRONMENT = 'Hockey-v2'
TEST_ENV_LEFT = 'HockeyLeft-v0'
TEST_ENV_MIDDLE = 'HockeyMiddle-v0'
TEST_ENV_RIGHT = 'HockeyRight-v0'
EPISODES = 100000
STEPS = 300
SAVE_RATE = 50
C_STEPS = 200			# (30 episodes)
BUFFER_FILE = 'Replay_buffer'
LOG_FILE = 'logs'
DISPLAY = False
DISPLAY_STATISTICS = False

##################
# counter
##################
log_counter = Counter("log")
save_counter = Counter("save")
Q_counter = Counter("Q_counter")
steps_counter = Counter("steps")
episodes_counter = Counter("episodes")
C_steps_counter = Counter("C_steps")

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

##################
# environments
##################
env = gym.make(ENVIRONMENT)
env_left = gym.make(TEST_ENV_LEFT)
env_middle = gym.make(TEST_ENV_MIDDLE)
env_right = gym.make(TEST_ENV_RIGHT)

##################
# graph auxiliries
##################
saver = tf.train.Saver()
init = tf.initialize_all_variables()
summary = tf.merge_all_summaries()

sess = tf.Session()

logger = tf.train.SummaryWriter(OUT_DIR, sess.graph)

# initialize variables (and target network)
sess.run(init)
Ws,bs = Q.get_weights()
Q_target.assign(sess, Ws,bs)
C_steps_counter.assign(sess,C_STEPS)
steps_counter.assign(sess, 0)
episodes_counter.assign(sess, 0)

ann_fric = (1-EPSILON)/ANNEALING
EXP_PROB = 1

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

# initialize logger
L = Logger()
log_not_empty = L.Load(OUT_DIR+LOG_FILE)
if log_not_empty:
	print ("Log file loaded")
else:
	("Creating new log file")
	L.AddNewLog('network_left')
	L.AddNewLog('network_middle')
	L.AddNewLog('network_right')
	L.AddNewLog('error')
	L.AddNewLog('total_reward')
	L.AddNewLog('estimated_value')
	L.AddNewLog('network_random')

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

################
# simulators
################
simulator = Simulator(STEPS, STATE_SIZE, FRAMES, T, actions_deque)

# main learning loop
print("Starting to learn in environment: " + ENVIRONMENT)
steps = steps_counter.evaluate(sess)
C_steps_counter.evaluate(sess)
for episode_i in xrange(1,EPISODES+1):
	episodes_counter.increment(sess)
	st = env.reset()
	mdp.add_frame(st)
	st = mdp.get_MDP_state()
	totalR = 0
	totalE = 0
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

		E_local = [0]
		if episode_i > OBSERVATION_PHASE:
			for mini_batch in xrange(BATCHES):
				# sample mini batch
				s_batch, a_batch, r_batch, stag_batch, terminal_batch,_ = R.SampleMiniBatch(MINI_BATCH)
				
				Y = Q.evaluate(sess, s_batch)				

				#Q_next_arg = Q.evaluate(sess, stag_batch)
				#Q_next_argmax = np.argmax(Q_next_arg,1)
				#Q_next_target = Q_target.evaluate(sess, stag_batch)

				#a_batch = a_batch.astype(int)
				#for i in range(MINI_BATCH):
				#	Y[i,a_batch[i,0]] = r_batch[i,0] + GAMMA*Q_next_target[i,Q_next_argmax[i]] * (1-terminal_batch[i])
				
				#error = Q.train(sess, s_batch, Y)

				# old DQN	
				Q_next = Q_target.evaluate(sess, stag_batch)
				Q_next_max = np.amax(Q_next,1)

				a_batch = a_batch.astype(int)
				for i in range(MINI_BATCH):
					Y[i,a_batch[i,0]] = r_batch[i,0] + GAMMA*Q_next_max[i] * (1-terminal_batch[i])
			
				# train on estimated Q next and rewards
				error = Q.train(sess, s_batch, Y)
				E_local.append(error)

		E_local = sum(E_local)/len(E_local)
		totalE += E_local

		if Done is True:
			break

	totalE = totalE/t		

	# run validation simulations
	L.AddRecord('network_left',simulator.SimulateNeuralEpisode(Q, sess, env_left, False))
	L.AddRecord('network_middle',simulator.SimulateNeuralEpisode(Q, sess, env_middle, False))
	L.AddRecord('network_right',simulator.SimulateNeuralEpisode(Q, sess, env_right, False))
	temp_r = 0
	for rand_i in xrange(10):
		temp_r = temp_r + simulator.SimulateNeuralEpisode(Q, sess, env, False) * 0.1
	L.AddRecord('network_random', temp_r)
	L.AddRecord('total_reward', totalR)
	L.AddRecord('error', totalE)
	s_est, _, _, _, _, num = R.SampleMiniBatch(V_EST)
	Q_est_arg = Q.evaluate(sess, s_est)
	Q_est_argmax = np.argmax(Q_est_arg,1)*1.0
	V_est = Q_est_argmax.sum()/num*1.0
	L.AddRecord('estimated_value', V_est)

	if steps >= C_STEPS:
		s,bs = Q.get_weights()
		Q_target.assign(sess, Ws,bs)
		print ('updating traget network')
		steps = 0
	steps += 1
	steps_counter.increment(sess)
		
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
		L.Save(OUT_DIR+LOG_FILE)	

Ws,bs = Q.get_weights()
Q_target.assign(sess, Ws,bs)
save_counter.increment(sess)
saver.save(sess,OUT_DIR+"model.ckpt", global_step=save_counter.evaluate(sess))
R.SaveBuffer(OUT_DIR+BUFFER_FILE)
L.Save(OUT_DIR+LOG_FILE)

sess.close()

# plot statistics
if DISPLAY_STATISTICS:
	R_Q_l = L.GetLogByName('network_left')
	R_Q_m = L.GetLogByName('network_middle')
	R_Q_r = L.GetLogByName('network_right')
	currentR = L.GetLogByName('network_random')
	error = L.GetLogByName('error')
	value = L.GetLogByName('estimated_value')
	
	t = np.arange(R_P_l.size)
	plt.figure()
	
	plt.subplot(231)
	plt.plot(t, R_Q_l, 'b')
	plt.xlabel('Episodes')
	plt.ylabel('reward')
	plt.title('Puck on the left')

	plt.subplot(232)
	plt.plot(t, R_Q_m, 'b')
	plt.xlabel('Episodes')
	plt.ylabel('reward')
	plt.title('Puck on the middle')

	plt.subplot(233)
	plt.plot(t, R_Q_r, 'b')
	plt.xlabel('Episodes')
	plt.ylabel('reward')
	plt.title('Puck on the right')
	
	plt.subplot(234)
	plt.plot(t, error, 'b')
	plt.xlabel('Episodes')
	plt.ylabel('error')
	plt.title('Learning error')

	plt.subplot(235)
	plt.plot(t, value, 'b')
	plt.xlabel('Episodes')
	plt.ylabel('value')
	plt.title('Estimated average value')
	
	plt.subplot(236)
	plt.plot(t, currentR, 'b')
	plt.xlabel('Episodes')
	plt.ylabel('reward')
	plt.title('Total reward')
		
	plt.suptitle('Simulation Statistics')	
	plt.show()

