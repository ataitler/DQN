import numpy as np
import tensorflow as tf
from utils.ReplayBuffer import ReplayBuffer
from utils.MDP_state import MDP_state
from collections import deque
import gym
import math
from utils.Policies import *
from utils.action_discretizer import *
from utils.Networks import *
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from utils.StatePreprocessor import *


# global definitions
OUT_DIR = "hockey_multinet1_decay_rate1.2_2_V6"
T = -0.5
Q_NET_SIZES = (40,100,100)
STATE_SIZE = 8
STATE_SIZE_POST = 8
FRAMES = 1
MDP_STATE_SIZE = FRAMES* STATE_SIZE_POST
ACTION_SIZE = 5
DISCRETIZATION = ACTION_SIZE ** 2
LEARNING_RATE = 0.001
GAMMA = 0.999
ANNEALING = 10000		# (5 episodes)
EPSILON = 0.1
L1W = 0.0
L2W = 0.0
BUFFER_SIZE = 200000
MINI_BATCH = 32
BATCHES = 1
OBSERVATION_PHASE = 2		# (2 episdoes)
#ENVIRONMENT = 'Hockey-v2'
ENVIRONMENT = 'HockeyLeft-v0'
EPISODES = 1
STEPS = 300
SAVE_RATE = 200
C_STEPS = 200			# (30 episodes)
BUFFER_FILE = 'Replay_buffer'
DISPLAY = True
DISPLAY_STATISTICS = True
DISPLAY_CONTROL = True
MONITOR = False

##################
# counter
##################
log_counter = Counter("log")
save_counter = Counter("save")

##################
# episode rewards
##################
reward = tf.placeholder(tf.float32, [1], name="reward")
save_reward = reward[0]
reward_sum = tf.scalar_summary('Reward',save_reward)

##################
# Q networks
##################
Q = DeterministicMLP("Q", STATE_SIZE_POST, Q_NET_SIZES, DISCRETIZATION, LEARNING_RATE, GAMMA, True)
Q_target1 = DeterministicMLP("Q_target1", STATE_SIZE_POST, Q_NET_SIZES, DISCRETIZATION, LEARNING_RATE, GAMMA, False,L2W,L1W)
#Q_target2 = DeterministicMLP("Q_target2", STATE_SIZE_POST, Q_NET_SIZES, DISCRETIZATION, LEARNING_RATE, GAMMA, False,L2W,L1W)
#Q_target3 = DeterministicMLP("Q_target3", STATE_SIZE_POST, Q_NET_SIZES, DISCRETIZATION, LEARNING_RATE, GAMMA, False,L2W,L1W)
#Q_target4 = DeterministicMLP("Q_target4", STATE_SIZE_POST, Q_NET_SIZES, DISCRETIZATION, LEARNING_RATE, GAMMA, False,L2W,L1W)
#Q_target5 = DeterministicMLP("Q_target5", STATE_SIZE_POST, Q_NET_SIZES, DISCRETIZATION, LEARNING_RATE, GAMMA, False,L2W,L1W)
#Q_target = DeterministicMLP("Q_target", STATE_SIZE_POST, Q_NET_SIZES, DISCRETIZATION, LEARNING_RATE, GAMMA, True)

saver = tf.train.Saver()
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

# initialize environment
env = gym.make(ENVIRONMENT)

# initialize mdp state structure
mdp = MDP_state(STATE_SIZE_POST, FRAMES)

# load saved model
ckpt = tf.train.get_checkpoint_state(OUT_DIR)
if ckpt and ckpt.model_checkpoint_path:
	saver.restore(sess,ckpt.model_checkpoint_path)
	print("Model loaded from disk")


# define action discretization
max_a = env.action_space.high[0]
min_a = env.action_space.low[0]
#table_max_x, table_max_y, malet_r = env.sizes()
#Processor = StatePreprocessor(STATE_SIZE, table_max_x*2, table_max_y*2, malet_r)

act = actions(ACTION_SIZE, max_a)
actions_deque,_ = act.get_action()
discretizer = Discretizer(actions_deque)

rewards = deque()
# main learning loop
print("Starting to learn in environment: " + ENVIRONMENT)
steps = 0
if MONITOR:
	env.monitor.start('movies/exp1')
for episode_i in xrange(1,EPISODES+1):
	st_pre = env.reset()
	st = st_pre
	mdp.reset()
	mdp.add_frame(st)
	st = mdp.get_MDP_state()
	state = np.array(st)
	control_effort = np.array([0,0]).reshape(1,2)
	Qs = np.array(Q.evaluate(sess,st)).reshape(1,DISCRETIZATION)
	totalR = 0
	for t in xrange(1,STEPS+1):
		if DISPLAY:
			env.render()

		# select action
		Q_vals = Q.evaluate(sess, st)
		a_index = np.argmax(Q_vals)
		at = actions_deque[a_index]
		
		# execute action
		st_next_pre, rt, Done, Done_act = env.step(at)
		st_next = st_next_pre
		mdp.add_frame(st_next)
		st_next = mdp.get_MDP_state()
		if Done_act:
			dt = 1
		else:
			dt = 0
		totalR += rt + T
		st = st_next
		#concat state and actions
		state = np.concatenate((state,st))
		control_effort = np.concatenate((control_effort,at.reshape(1,2)))
		Qv = Q_vals.reshape(1,DISCRETIZATION)
		Qmax = np.max(Qv)
		Qmin = np.min(Qv)
		#Qv = (Qv-Qmin)/(Qmax-Qmin)
		Qs = np.concatenate((Qs,Qv))
		

		if Done is True:
			if DISPLAY:
				env.render()
			break

	#while not Done:
		#if DISPLAY:
		#	env.render()
		#_,_,Done,_ = env.step(at)

	print "finished episode " , episode_i
	rewards.append(totalR)

if DISPLAY_STATISTICS:
	R = np.array(rewards)
	avg = 1.*R.sum()/R.size
	var = np.sqrt(np.power(R-avg,2).sum()/(R.size-1))
	print "Average reward: ", avg, " variance: ", var

	R = np.array(rewards)	
	plt.plot(R)
	plt.title(u'\u03bc = ' + str(avg) + u', \u03c3 = ' + str(var))
	plt.xlabel('episodes')
	plt.ylabel('reward')
	plt.show()


if DISPLAY_CONTROL:
	Qs = np.delete(Qs,0,0)
	X = np.arange(1,Qs.shape[0]+1)
	Y = np.arange(1,Qs.shape[1]+1)
	X,Y = np.meshgrid(Y,X)
	
	control_effort = np.delete(control_effort,0,0)
	t_max = state.shape[0]
	state = np.delete(state,t_max-1,0)
	t = np.arange(t_max-1)
	fig = plt.figure(1)
	plt.subplot(3,2,1)
	plt.plot(t,state[:,0]) # position X
	plt.title('Y position')
	plt.xlabel('Time [steps]')
	plt.ylabel('Position')
	plt.ylim(-20,-5)
	plt.xlim(0,t_max-1)
	plt.subplot(3,2,3)
	plt.plot(t,state[:,2]) # velocity X
	plt.title('Y velocity')
	plt.ylabel('Velocity')
	plt.xlabel('Time [steps]')
	plt.ylim(-11,11)
	plt.xlim(0,t_max-1)
	plt.subplot(3,2,5)
	plt.step(t,control_effort[:,0]) # control X
	plt.title('Y control signal')
	plt.xlabel('Time [steps]')
	plt.ylabel('Acceleration')
	plt.ylim(-11,11)
	plt.xlim(0,t_max-1)
	plt.subplot(3,2,2)
	plt.plot(t,state[:,1]) # position Y
	plt.title('X position')
	plt.xlabel('Time [steps]')
	plt.ylabel('Position')
	plt.ylim(-10,10)
	plt.xlim(0,t_max-1)
	plt.subplot(3,2,4)
	plt.plot(t,state[:,3]) # velocity Y
	plt.title('X velocity')
	plt.xlabel('Time [steps]')
	plt.ylabel('Velocity')
	plt.ylim(-11,11)
	plt.xlim(0,t_max-1)
	plt.subplot(3,2,6)
	plt.step(t,control_effort[:,1]) # control Y
	plt.title('X control signal')
	plt.xlabel('Time [steps]')
	plt.ylabel('Acceleration')
	plt.ylim(-11,11)
	plt.xlim(0,t_max-1)
	
	fig2 = plt.figure(2)
	plt.plot(state[:,1],state[:,0])
	#plt.xlim(-5,1)
	#plt.xlim(-2,2)
	#plt.xlim(-1,5)
	#plt.ylim(-18,-7)
	plt.title('The agent\'s trajectory in the table plane')
	plt.xlim(-5,5)
	plt.xlabel('X')
	plt.ylabel('Y')

	fig3 = plt.figure(3)
	ax = fig3.gca(projection='3d')
	surf = ax.plot_surface(X, Y, Qs, cmap=cm.coolwarm)
	ax.set_xlabel('Action index')	
	ax.set_ylabel('Step')
	ax.set_zlabel('Q values')
	ax.set_title('Q values Vs step number')
	fig3.colorbar(surf, shrink=0.5, aspect=5)

	plt.show()


	
		
sess.close()
if MONITOR:
	env.monitor.close()

