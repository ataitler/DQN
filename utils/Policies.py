import numpy as np
from collections import deque
from action_discretizer import *

class ChasePolicy():
	dt = 0.05

	def __init__(self, state_size, action_size, max_a, min_a):
		self.state_size = state_size
		self.action_size = action_size
		self.max_a = max_a
		self.min_a = min_a

	def action(self, state):
		rows = len(state)
		actions = deque(maxlen = rows)
		for i in xrange(rows):
			# velocity cancelation
			# force aiming towards the puck
			Vx_next = state[i,4] - state[i,0]
			Vy_next = state[i,5] - state[i,1]
			n = self._norm(Vx_next, Vy_next)
			if n == 0:
				X1 = -state[i,2]/self.dt
				X2 = -state[i,3]/self.dt
			else:
				X1 = (Vx_next/n*self.max_a - state[i,2])/self.dt
				X2 = (Vy_next/n*self.max_a - state[i,3])/self.dt
			nx = self._norm(X1, X2)
			if nx == 0:
				a = np.array([0, 0])
			else:
				a = np.array([X1, X2])/nx * self.max_a

			# action is combination of the two componants
			actions.append(a)
		action = np.array(actions)
		action = action.reshape(rows, 2)
		return action

	def _norm(self, X1, X2):
		return np.sqrt( X1**2 + X2**2 )


class Potential():
	def __init__(self, action_num, max_a, max_v):
		self.Qsize = action_num**2
		act = actions(action_num, max_a)
		self.actions,_ = act.get_action()
		self.discretizer = Discretizer(self.actions)

		self.max_force = max_a
		self.max_speed = max_v
		self.dt = 0.5

	def GetQ(self, st):
		Q = np.zeros(self.Qsize)
		for i in xrange(self.Qsize):
			at = self._index2action(i)
			#Q[i] = self._stateBaseEval(at,st)
			Q[i] = self._actionBaseEval(at,st)
		return Q.reshape(1,self.Qsize)

	def _calcDynamics(self, state, action):
		aVx = np.clip(state[0,2] + self.dt*action[0], -self.max_speed, self.max_speed)
		aVy = np.clip(state[0,3] + self.dt*action[1], -self.max_speed, self.max_speed)
		aX = round(state[0,0] + self.dt * aVx)
		aY = round(state[0,1] + self.dt * aVy)

		pVx = np.clip(state[0,6], -self.max_speed, self.max_speed)
		pVy = np.clip(state[0,7], -self.max_speed, self.max_speed)
		pX = round(state[0,4] + self.dt * pVx)
		pY = round(state[0,5] + self.dt * pVy)

		# need to flip puck velocity in y axis in case of wall hit

		stag = np.array([[aX, aY, aVx, aVy, pX, pY, pVx, pVy]])
		return stag

	def _index2action(self, index):
		action = self.actions[index]
		return action

	def _stateBaseEval(self, action, state):
		stag = self._calcDynamics(state, action)
		return self._evalState(stag)

	def _actionBaseEval(self, action, state):
		a_best = self._calcBestAction(state)
		return self._evalAction(action, a_best)

	def _evalState(self, stag):
		eps = 0.001
		dist = np.sqrt( (stag[0,0]-stag[0,4])**2 + (stag[0,1]-stag[0,5])**2 )
		return 1.0/(dist + eps)

	def _calcBestAction(self, st):
		Vx_next = st[0,4] - st[0,0]
		Vy_next = st[0,5] - st[0,1]
		n = self._norm(Vx_next, Vy_next)
		if n == 0:
			X1 = -st[0,2]/self.dt
			X2 = -st[0,3]/self.dt
		else:
			X1 = (Vx_next/n*self.max_force - st[0,2])/self.dt
			X2 = (Vy_next/n*self.max_force - st[0,3])/self.dt
		nx = self._norm(X1, X2)
		if nx == 0:
			at = np.array([0, 0])
		else:
			at = np.array([X1, X2])/nx * self.max_force

		_,_,a_index = self.discretizer.get_discrete(at.reshape(1,2))
		return self.actions[a_index[0]]

	def _evalAction(self, at, at_best):
		at_norm = self._norm(at[0],at[1])
		if at_norm == 0:
			at = np.array([0, 0])
		else:
 			at = at/at_norm
		at_best_norm = self._norm(at_best[0], at_best[1])
		if at_best_norm == 0:
			at_best = np.array([0, 0])
		else:
			at_best = at_best/at_best_norm
		return np.dot(at, at_best)

	def _norm(self, X1, X2):
		return np.sqrt( X1**2 + X2**2 )


