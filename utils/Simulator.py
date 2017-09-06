import numpy as np
import tensorflow as tf
from utils.MDP_state import MDP_state
from collections import deque
import gym
import math
from utils.Networks import *
from utils.action_discretizer import *


class Simulator():
	def __init__(self, steps, state_size, frames, T, actions):
		self.steps = steps
		self.frames = frames
		self.mdp = MDP_state(state_size, frames)
		self.T = T
		self.actions = actions

	def SimulateContNeuralEpisode(self, Actor, sess, env, display):
		st = env.reset()
		self.mdp.reset()	
		for i in xrange(self.frames):
			self.mdp.add_frame(st)
		st = self.mdp.get_MDP_state()
		totalR = 0
		for step in xrange(self.steps):
			if display:
				env.render()
			at = Actor.predict(st)
			st_next,rt,Done,_ = env.step(at[0])
			rt += self.T
			self.mdp.add_frame(st_next)
			st_next = self.mdp.get_MDP_state()
			if Done:
				dt = 1
			else:
				dt = 0
			totalR += rt
			st = st_next
			if Done is True:
				break
		return totalR

	def SimulateNeuralEpisode(self, Q, sess, env, display):
		st = env.reset()
		self.mdp.reset()
		for i in xrange(self.frames):
			self.mdp.add_frame(st)
		st = self.mdp.get_MDP_state()
		totalR = 0
		for step in xrange(self.steps):
			if display:
				env.render()
			q_vector = Q.evaluate(sess, st)
			a_index = np.argmax(q_vector)
			at = self.actions[a_index]
			st_next, rt, Done,_ = env.step(at)
			rt += self.T
			self.mdp.add_frame(st_next)
			st_next = self.mdp.get_MDP_state()
			if Done:
				dt = 1
			else:
				dt = 0
			totalR += rt
			st = st_next
			if Done is True:
				break
		return totalR
	
	def SimulatePolicyEpisode(self, policy, discretizer, env, display):
		st = env.reset()
		self.mdp.reset()
		for i in xrange(self.frames):
			self.mdp.add_frame(st)
		st = self.mdp.get_MDP_state()
		totalR = 0
		for step in xrange(self.steps):
			if display:
				env.render()
			action_cont = policy.action(st)
			_,_,a_index = discretizer.get_discrete(action_cont)
			at = self.actions[a_index[0]]
			st_next, rt, Done,_ = env.step(at)
			rt += self.T
			self.mdp.add_frame(st_next)
			st_next = self.mdp.get_MDP_state()
			if Done:
				dt = 1
			else:
				dt = 0
			totalR += rt
			st = st_next
			if Done is True:
				break
		return totalR
		


