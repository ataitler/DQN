import numpy as np
from collections import deque

class MDP_state():
	def __init__(self, frame_size, num_frames):
		self.frame_size = frame_size
		self.num_frames = num_frames
		self.augmented_state_size = frame_size * num_frames
		self.state = deque(maxlen = num_frames)
		for i in xrange(num_frames):
			self.state.append(np.zeros(frame_size))

	def add_frame(self, frame):
		self.state.popleft()
		self.state.append(frame)

	def get_MDP_state(self):
		temp = deque(self.state)
		temp.reverse()
		mdp_s = np.array(temp)
		mdp_s = mdp_s.reshape(1,self.augmented_state_size)
		return mdp_s

	def reset(self):
		# initialize
		for i in xrange(self.num_frames):
			self.state.append(np.zeros(self.frame_size))

