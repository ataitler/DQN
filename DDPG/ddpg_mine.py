import numpy as np
import tensorflow as tf
from utils.ReplayBuffer import ReplayBuffer
from utils.MDP_state import MDP_state
from collections import deque
import gym
import math
from utils.Networks import *
#from utils.Policies import *
#from utils.StatePreprocessor import *
from utils.action_discretizer import *
from utils.OUnoise import *
import sys

# global definitions
OUT_DIR = "test1/"
T = -0.5
Q_NET_SIZES = (400,300)
P_NET_SIZES = (400,300)
STATE_SIZE = 3
FRAMES = 1
MDP_STATE_SIZE = FRAMES* STATE_SIZE
ACTION_SIZE = 1
ACTION_DIM = 1
DISCRETIZATION = ACTION_SIZE ** ACTION_DIM
LEARNING_RATE = 0.0001
GAMMA = 0.999
TAU = 0.001
ANNEALING = 10000			# (35 episodes)
EPSILON = 0.1
BUFFER_SIZE = 10000
MINI_BATCH = 128
BATCHES = 1
OBSERVATION_PHASE = 2		# (2 episdoes)
ENVIRONMENT = 'Pendulum-v0'
EPISODES = 400
STEPS = 100
SAVE_RATE = 200
C_STEPS = 200
L2A = 0
L2C = 0.0	
BUFFER_FILE = 'Replay_buffer'
DISPLAY = False
SUMMARIES = False

env = gym.make(ENVIRONMENT)

###################
# counter`
###################
log_counter = Counter("log")
save_counter = Counter("save")

##################
# episode rewards
##################
reward = tf.placeholder(tf.float32, [1], name="reward")
save_reward = reward[0]
reward_sum = tf.scalar_summary('Reward',save_reward)

sess = tf.InteractiveSession()

##################
# networks
##################
Actor = ActorNetwork(sess, MDP_STATE_SIZE, ACTION_SIZE, Q_NET_SIZES, MINI_BATCH, TAU, 0.0001, L2A)
Critic = CriticNetwork(sess, MDP_STATE_SIZE, ACTION_SIZE, P_NET_SIZES, MINI_BATCH, TAU, 0.001, L2C)

##################
# graph auxiliries
##################
saver = tf.train.Saver()
init = tf.initialize_all_variables()
summary = tf.merge_all_summaries()

logger = tf.train.SummaryWriter(OUT_DIR, sess.graph)

# initialize mdp state structure
mdp = MDP_state(STATE_SIZE, FRAMES)

# initialize replay buffer
R = ReplayBuffer(MDP_STATE_SIZE, ACTION_SIZE, BUFFER_SIZE)
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
#max_a = env.action_space.high[0]
#min_a = env.action_space.low[0]

n = OUnoise(1,0.5,1)

# main learning loop
print("Starting to learn in environment: " + ENVIRONMENT)
steps = 0
for episode_i in xrange(1,EPISODES+1):
	st = env.reset()
	n.Reset()
	mdp.reset()
	for i in xrange(FRAMES):
		mdp.add_frame(st)
	st = mdp.get_MDP_state()
	totalR = 0
	for t in xrange(1,STEPS+1):
		if DISPLAY:
			env.render()
		at = Actor.predict(st) + n.Sample()
		# execute action
		st_next, rt, Done, _ = env.step(at[0])
		mdp.add_frame(st_next)
		st_next = mdp.get_MDP_state()
		if Done:
			dt = 1
		else:
			dt = 0
		totalR += rt

		# store transition
		R.StoreTransition(st, at, np.array([rt]), st_next, dt)
		st = st_next

		if episode_i > OBSERVATION_PHASE:
			for mini_batch in xrange(BATCHES):
				# sample mini batch
				s_batch, a_batch, r_batch, stag_batch, terminal_batch,_ = R.SampleMiniBatch(MINI_BATCH)
				
				Q_next = Critic.target_predict(stag_batch, Actor.target_predict(stag_batch))
				Y = r_batch + GAMMA*Q_next * (1-terminal_batch)

				Critic.train(Y, s_batch, a_batch)
				
				a_for_grad = Actor.predict(s_batch)
				grads = Critic.gradients(s_batch, a_batch)
				Actor.train(s_batch, grads)
				
				Actor.target_train()
				Critic.target_train()

		if Done is True:
			break

	r = np.array([totalR])
	log = sess.run(summary,{reward:r})
	log_counter.increment(sess)
	current = log_counter.evaluate(sess)
	logger.add_summary(log,current)
	print ("episode %d/%d (%d), reward: %f" % (episode_i, EPISODES, current, totalR))

save_counter.increment(sess)
saver.save(sess,OUT_DIR+"model.ckpt", global_step=save_counter.evaluate(sess))
R.SaveBuffer(OUT_DIR+BUFFER_FILE)

sess.close()
