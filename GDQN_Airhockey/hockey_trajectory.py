import numpy as np
import tensorflow as tf
from utils.ReplayBuffer import *
from utils.Logger import Logger
from utils.MDP_state import MDP_state
from collections import deque
import gym
import math
from utils.Networks import *
from utils.Policies import *
from utils.StatePreprocessor import *
from utils.action_discretizer import *
from utils.OUnoise import *
from utils.Simulator import *
import matplotlib.pyplot as plt
import sys

# global definitions
OUT_DIR = "hockey_trajupdate/"
T = -0.5
Q_NET_SIZES = (40,100,100)
STATE_SIZE = 8
STATE_SIZE_POST = 8
FRAMES = 1
MDP_STATE_SIZE = FRAMES* STATE_SIZE_POST
ACTION_SIZE = 5
ACTION_DIM = 2
DISCRETIZATION = ACTION_SIZE ** ACTION_DIM
NOISE = 100
LEARNING_RATE = 0.0001
GAMMA = 0.999
L2W = 0.1
L1W = 0.0
ANNEALING = 10000		# (35 episodes)
EPSILON_P = 0.1
EPSILON = 0.1
EPSILON_T = 0.5
BUFFER_SIZE = 100000
MINI_BATCH = 32
V_EST = 500
BATCHES = 2
OBSERVATION_PHASE = 1		# (2 episdoes)
ENVIRONMENT = 'Hockey-v2'
TEST_ENV_LEFT = 'HockeyLeft-v0'
TEST_ENV_MIDDLE = 'HockeyMiddle-v0'
TEST_ENV_RIGHT = 'HockeyRight-v0'
EPISODES = 2000
STEPS = 300
SAVE_RATE = 100
C_STEPS = 100
BUFFER_FILE = 'Replay_buffer'
BUFFER_VAL = 'Replay_val'
LOG_FILE = 'logs'
DISPLAY = False
NEW_EXPERIENCE = True
ONLY_OUTPUT = False

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
Q = DeterministicMLP("Q", STATE_SIZE_POST, Q_NET_SIZES, DISCRETIZATION, LEARNING_RATE, GAMMA, True,L2W,L1W)
Q_target = DeterministicMLP("Q_target", STATE_SIZE_POST, Q_NET_SIZES, DISCRETIZATION, LEARNING_RATE, GAMMA, True,L2W,L1W)

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

tflogger = tf.train.SummaryWriter(OUT_DIR, sess.graph)

# initialize variables (and target network)
sess.run(init)
Ws,bs = Q.get_weights()
Q_target.assign(sess, Ws,bs)

ann_fric = (1-EPSILON)/ANNEALING
EXP_PROB = 1

# initialize mdp state structure
mdp = MDP_state(STATE_SIZE_POST, FRAMES)

# initialize replay buffer
R_val = ReplayBuffer(MDP_STATE_SIZE, 1, BUFFER_SIZE)
R = TrajectoryReplayBuffer(MDP_STATE_SIZE, 1, BUFFER_SIZE, STEPS)
buf = R.LoadBuffer(OUT_DIR+BUFFER_FILE)
buf_val = R_val.LoadBuffer(OUT_DIR+BUFFER_VAL)
if buf:
	OBSERVATION_PHASE = 0
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
	L.AddNewLog('policy_left')
	L.AddNewLog('policy_middle')
	L.AddNewLog('policy_right')
	L.AddNewLog('error')
	L.AddNewLog('total_reward')
	L.AddNewLog('estimated_value')

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
policy = ChasePolicy(STATE_SIZE, ACTION_SIZE, max_a, min_a)
n = OUnoise(2,0.5,NOISE)

################
# simulators
################
simulator = Simulator(STEPS, STATE_SIZE, FRAMES, T, actions_deque)

# main learning loop
print("Starting to learn in environment: " + ENVIRONMENT)
steps = 0
for episode_i in xrange(1,EPISODES+1):

	policy_exp = np.random.uniform()
	n.Reset()
	if policy_exp <= EPSILON_P:
		onPolicy = True
	else:
		onPolicy = False
	
	st = env.reset()
	mdp.reset() 
	mdp.add_frame(st)
	st = mdp.get_MDP_state()
	totalR = 0
	totalE = 0
	for t in xrange(1,STEPS+1):
		if DISPLAY:
			env.render()

		# select action
		if onPolicy:
			# pick action from policy
			action_pre = policy.action(st) 
			at_pi_cont = action_pre 
			_,_,a_index = discretizer.get_discrete(at_pi_cont)
			a_index = a_index[0]
		else:
			action_exp = np.random.uniform()
			if action_exp > EXP_PROB:
				q_vector = Q.evaluate(sess, st)
				a_index = np.argmax(q_vector)
				noise = n.Sample()
				cont_action = np.array([actions_deque[a_index] + noise])
				_,_,a_index = discretizer.get_discrete(cont_action)
				a_index = a_index[0]
			else:
				a_index = np.random.randint(0,DISCRETIZATION)

			# reduce exploration probability
			if EXP_PROB > EPSILON:
				EXP_PROB -= ann_fric
		
		at = actions_deque[a_index]
		
		# execute action
		st_next, rt, Done, _ = env.step(at)
		rt += T
		mdp.add_frame(st_next)
		st_next = mdp.get_MDP_state()	
		if Done:
			dt = 1
		else:
			dt = 0
		totalR += rt
		
		# store transition
		if NEW_EXPERIENCE:
			R.AddTransition(st, np.array([a_index]), np.array([rt]), st_next, dt)
			R_val.StoreTransition(st, np.array([a_index]), np.array([rt]), st_next, dt)
		st = st_next
		
		E_local=[0]
		if episode_i > OBSERVATION_PHASE:
			E_local=[]
			for mini_batch in xrange(BATCHES):
				# sample mini batch
				#s_batch, a_batch, r_batch, stag_batch, terminal_batch, num = R.SampleMiniBatch(MINI_BATCH)
				traj_prob = np.random.uniform()
				if traj_prob > EPSILON_T:
					s_batch, a_batch, r_batch, stag_batch, terminal_batch = R.GetLastTrajectory()
				else:
					s_batch, a_batch, r_batch, stag_batch, terminal_batch = R.SampleTrajectory()
				num = s_batch.shape[0]
				Y = Q.evaluate(sess, s_batch)
			
				# Double DQN update	
				#Q_next_arg = Q.evaluate(sess, stag_batch)
				#Q_next_argmax = np.argmax(Q_next_arg,1)
				#Q_next_target = Q_target.evaluate(sess, stag_batch)

				#a_batch = a_batch.astype(int)
				#for i in range(num):
				#	Y[i,a_batch[i,0]] = r_batch[i,0] + GAMMA*Q_next_target[i,Q_next_argmax[i]] * (1-terminal_batch[i])

				#if ONLY_OUTPUT:
				#	error = Q.train_output(sess, s_batch, Y)
				#else:
				#	error = Q.train(sess, s_batch, Y)
				#E_local.append(error)
				
				# Standard DQN update
				Q_next = Q_target.evaluate(sess, stag_batch)
				Q_next_max = np.amax(Q_next,1)

				a_batch = a_batch.astype(int)
				for i in range(num):
					Y[i,a_batch[i,0]] = r_batch[i,0] + GAMMA*Q_next_max[i] * (1-terminal_batch[i])

				# train on estimated Q next and rewards
				if ONLY_OUTPUT:
					error = Q.train_output(sess, s_batch, Y)
				else:
					error = Q.train(sess, s_batch, Y)
				E_local.append(error)

		E_local = sum(E_local)/len(E_local)
		totalE += E_local

		if Done is True:
			break

	totalE = totalE/t
	R.FinilizeTrajectory()

	# run validation simulations
	a = simulator.SimulateNeuralEpisode(Q, sess, env_left, False)
	L.AddRecord('network_left', a)
	L.AddRecord('network_middle',simulator.SimulateNeuralEpisode(Q, sess, env_middle, False))
	L.AddRecord('network_right',simulator.SimulateNeuralEpisode(Q, sess, env_right, False))
	L.AddRecord('policy_left',simulator.SimulatePolicyEpisode(policy,discretizer, env_left, False))
	L.AddRecord('policy_middle',simulator.SimulatePolicyEpisode(policy,discretizer, env_middle, False))
	L.AddRecord('policy_right',simulator.SimulatePolicyEpisode(policy,discretizer, env_right, False))
	L.AddRecord('total_reward', totalR)
	L.AddRecord('error', totalE)
	s_est, _, _, _, _, num = R_val.SampleMiniBatch(V_EST)
	Q_est_arg = Q.evaluate(sess, s_est)
	Q_est_argmax = np.argmax(Q_est_arg,1)*1.0
	V_est = Q_est_argmax.sum()/num*1.0
	L.AddRecord('estimated_value', V_est)
	
	# update target network
	if steps >= C_STEPS:
		Ws,bs = Q.get_weights()
		Q_target.assign(sess, Ws,bs)
		print ('updating traget network')
		steps = 0
	steps += 1
	
	# update reward log
	if onPolicy == False:	
		r = np.array([totalR])
		log = sess.run(summary,{reward:r})
		log_counter.increment(sess)
		tflogger.add_summary(log,log_counter.evaluate(sess))
		print ("episode %d/%d (%d), reward: %f" % (episode_i, EPISODES, log_counter.evaluate(sess), totalR))
	else:
		print ("episode %d/%d, played on policy, reward: %f" % (episode_i, EPISODES, totalR))

	# save model
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
print "model saved, replay buffer: ", R.GetOccupency()
L.Save(OUT_DIR+LOG_FILE)

sess.close()
sys.exit()
# plot statistics
R_P_l = L.GetLogByName('policy_left')
R_Q_l = L.GetLogByName('network_left')
R_P_m = L.GetLogByName('policy_middle')
R_Q_m = L.GetLogByName('network_middle')
R_P_r = L.GetLogByName('policy_right')
R_Q_r = L.GetLogByName('network_right')
totalR = L.GetLogByName('total_reward')
error = L.GetLogByName('error')
value = L.GetLogByName('estimated_value')

t = np.arange(R_P_l.size)
plt.figure(1)
plt.plot(t, R_Q_l, 'b', t, R_P_l, 'r')
plt.xlabel('Episodes')
plt.ylabel('reward')
plt.title('Puck on the left')

plt.figure(2)
plt.plot(t, R_Q_m, 'b', t, R_P_m, 'r')
plt.xlabel('Episodes')
plt.ylabel('reward')
plt.title('Puck on the middle')

plt.figure(3)
plt.plot(t, R_Q_r, 'b', t, R_P_r, 'r')
plt.xlabel('Episodes')
plt.ylabel('reward')
plt.title('Puck on the right')

plt.figure(4)
plt.plot(t, totalR, 'b')
plt.xlabel('Episodes')
plt.ylabel('reward')
plt.title('Total reward')

plt.figure(5)
plt.plot(t, error, 'b')
plt.xlabel('Episodes')
plt.ylabel('error')
plt.title('Learning error')

plt.figure(6)
plt.plot(t, value, 'b')
plt.xlabel('Episodes')
plt.ylabel('value')
plt.title('Estimated average value')

plt.show()
	

