import numpy as np
from collections import deque


class actions():
	def __init__(self, d, limit):
		ax = np.linspace(-limit,limit,d)
		ay = np.linspace(-limit,limit,d)
		mlen = np.power(d,2)
		self.a = deque(maxlen=mlen)
		for i in xrange(d):
			for j in xrange(d):
				self.a.append(np.array([ax[i],ay[j]]))
		self.aa = np.array(self.a)

	def get_action(self):
		return self.a, self.aa

class Discretizer():
	def __init__(self,d_array):
		self.values = np.array(d_array)
		self.size = len(d_array)
		self.dims = len(d_array[0])
		self.vals_in_axis = np.power(self.size, 1./self.dims)

		self.min_val = d_array[0][0]
		self.max_val = d_array[-1][0]
		
		self.units = (self.max_val-self.min_val) / (self.vals_in_axis-1)

	def get_discrete(self, cont_val):
		cont_val = np.clip(cont_val, self.min_val, self.max_val)
		states = len(cont_val)
		cont_val = cont_val + np.absolute(self.min_val)
		div = np.floor(cont_val / self.units).astype(int)
		mod = cont_val % self.units
		for j in xrange(states):
			for i in xrange(self.dims):
				if mod[j][i] > self.units/2.:
					div[j][i] += 1
		index = (div[:,0]*self.vals_in_axis+div[:,1]).astype(int)
		one_hot_action = np.zeros((states, self.size))
		action = np.zeros((states, self.dims))
		try:
			for i in xrange(states):
				one_hot_action[i][index[i]] = 1
				action[i,:] = self.values[index[i]]
		except:
			print index
		return  action, one_hot_action, index
		
		
#ac = actions(11,20)
#a,aa = ac.get_action()
#d = Discretizer(a)
#to_d = np.array([-1.9,20])
#print to_d
#div,action = d.get_discrete(to_d)
#print action

