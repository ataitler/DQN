import numpy as np
import tensorflow as tf
import gym
from ReplayBuffer import ReplayBuffer
from ActorNetwork import ActorNetwork
from CriticNetwork import CriticNetwork
from utils.Simulator import *
from utils.Logger import Logger
from utils.Networks import Counter
import timeit
import sys
from utils.OUnoise import OUnoise

# REPLAY BUFFER CONSTS
T = -0.5
BUFFER_SIZE = 200000
BATCH_SIZE = 64
# FUTURE REWARD DECAY
GAMMA = 0.999
# TARGET NETWORK UPDATE STEP
TAU = 0.001
# LEARNING_RATE
V_EST = 500
LRA = 0.0001
LRC = 0.001
#ENVIRONMENT_NAME
#ENVIRONMENT_NAME = 'Pendulum-v0'
ENVIRONMENT_NAME = 'Hockey-v2'
TEST_ENV_LEFT = 'HockeyLeft-v0'
TEST_ENV_MIDDLE = 'HockeyMiddle-v0'
TEST_ENV_RIGHT = 'HockeyRight-v0'
# L2 REGULARISATION
L2C = 0.01
L2A = 0.0
if ENVIRONMENT_NAME is 'Hockey-v2':
	STEPS = 300
else:
	STEPS = 100
EPISODES = 100000
SAVE_RATE = 50
STATE_SIZE = 8
FRAMES = 1
NOISE = 1
log_counter = 0
OUT_DIR="hockey_DDPG_001_3/"
LOG_FILE = 'logs'
DISPLAY = False

##################
# counter
##################
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


env = gym.make(ENVIRONMENT_NAME)
if ENVIRONMENT_NAME is 'Hockey-v2':
	env_left = gym.make(TEST_ENV_LEFT)
	env_middle = gym.make(TEST_ENV_MIDDLE)
	env_right = gym.make(TEST_ENV_RIGHT)
action_dim = env.action_space.shape[0]
action_high = env.action_space.high
action_low = env.action_space.low

input_dim = env.observation_space.shape[0]

sess = tf.InteractiveSession()
logger = tf.train.SummaryWriter(OUT_DIR, sess.graph)
actor = ActorNetwork(sess, input_dim, action_dim, BATCH_SIZE, TAU, LRA, L2A)
critic = CriticNetwork(sess, input_dim, action_dim, BATCH_SIZE, TAU, LRC, L2C)
buff = ReplayBuffer(BUFFER_SIZE)
summary = tf.merge_all_summaries()

n = OUnoise(action_dim, 0.15, NOISE)
#n = OUnoise(action_dim)

saver = tf.train.Saver()
ckpt = tf.train.get_checkpoint_state(OUT_DIR)
if ckpt and ckpt.model_checkpoint_path:
	saver.restore(sess,ckpt.model_checkpoint_path)
	print("Model loaded from disk")

# initialize logger
L = Logger()
log_not_empty = L.Load(OUT_DIR+LOG_FILE)
if log_not_empty:
	print ("Log file loaded")
else:
	("Creating new log file")
	if ENVIRONMENT_NAME is 'Hockey-v2':
		L.AddNewLog('network_left')
		L.AddNewLog('network_middle')
		L.AddNewLog('network_right')
		L.AddNewLog('network_random')
#	L.AddNewLog('error')
	L.AddNewLog('total_reward')
	L.AddNewLog('estimated_value')
	L.AddNewLog('network_random')

if ENVIRONMENT_NAME is 'Hockey-v2':
	simulator = Simulator(STEPS, STATE_SIZE, FRAMES, T, None)
steps = steps_counter.evaluate(sess)
C_steps_counter.evaluate(sess)
for ep in range(EPISODES):
    episodes_counter.increment(sess)
    # open up a game state
    s_t, r_0, done = env.reset(), 0, False
    n.Reset()
    REWARD = 0
    totalR = 0
    totalE = 0
    # exploration.reset()
    for t in range(STEPS):
	if DISPLAY:
	        env.render()
        # select action according to current policy and exploration noise
	noise = n.Sample()
	#print noise
        a_t = actor.predict([s_t]) + noise

        # execute action and observe reward and new state
        s_t1, r_t, done, info = env.step(a_t[0])
	r_t = r_t+T
	if done:
		dt = 1
	else:
		dt = 0

        # store transition in replay buffer
        buff.add(s_t, a_t[0], r_t, s_t1, dt)
        # sample a random minibatch of N transitions (si, ai, ri, si+1) from replay buffer
        batch = buff.getBatch(BATCH_SIZE)
        states = np.asarray([e[0] for e in batch])
        actions = np.asarray([e[1] for e in batch])
        rewards = np.asarray([e[2] for e in batch])
        new_states = np.asarray([e[3] for e in batch])
        dones = np.asarray([e[4] for e in batch])
        
	target_q_values = critic.target_predict(new_states, actor.target_predict(new_states))
	rt = rewards.reshape(rewards.size,1)
	dones = dones.reshape(dones.size,1)
        y_t = rt + GAMMA*target_q_values*(1-dones)
        
	# update critic network by minimizing los L = 1/N sum(yi - critic_network(si,ai))**2
        critic.train(y_t, states, actions)

        # update actor policy using sampled policy gradient
        a_for_grad = actor.predict(states)
        grads = critic.gradients(states, a_for_grad)
        actor.train(states, grads)

        # update the target networks
        actor.target_train()
        critic.target_train()

        # move to next state
        s_t = s_t1
        REWARD += r_t
	totalR += r_t
	if done == True:
		break

    #update statistics
    if ENVIRONMENT_NAME is 'Hockey-v2':
	    L.AddRecord('network_left',simulator.SimulateContNeuralEpisode(actor, sess, env_left, False))
	    L.AddRecord('network_middle',simulator.SimulateContNeuralEpisode(actor, sess, env_middle, False))
	    L.AddRecord('network_right',simulator.SimulateContNeuralEpisode(actor, sess, env_right, False))
	    temp_r = 0
	    for rand_i in xrange(10):
	    	temp_r = temp_r + simulator.SimulateContNeuralEpisode(actor, sess, env, False) * 0.1
	    L.AddRecord('network_random', temp_r)
    L.AddRecord('total_reward', totalR)
    batch = buff.getBatch(V_EST)
    s_est = np.asarray([e[0] for e in batch])
    num = s_est.shape[0]
    Q_est = critic.predict(s_est, actor.predict(s_est))
    V_est = Q_est.sum()/num*1.0
    L.AddRecord('estimated_value', V_est)

    print "EPISODE ", ep, "ENDED UP WITH REWARD: ", REWARD
    r = np.array([REWARD])
    log = sess.run(summary,{reward:r})
    log_counter+=1
    logger.add_summary(log,log_counter)

    if ep % SAVE_RATE == 0:
        print "model saved, buffer size: ", buff.size()
    	save_counter.increment(sess)
    	saver.save(sess,OUT_DIR+"model.ckpt")
        L.Save(OUT_DIR+LOG_FILE)
	

saver.save(sess,OUT_DIR+"model.ckpt")
