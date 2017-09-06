import numpy as np
from collections import deque

class StatePreprocessor():
	def __init__(self, state_size, number_of_frames, table_width, table_height, puck_radius, agent_radius, discretization = 0.1):
		self.d = discretization
		self.number_of_frames = number_of_frames
		self.state_size = state_size
		self.table_height = table_height/2.0
		self.table_width = table_width * 1.0
		self.puck_r = puck_radius
		self.agent_r = agent_radius

		self.rows_sparse = int(self.table_height / discretization)
		#self.cols = int(self.table_width / discretization)
		#self.pixels = self.rows * self.cols
		#self.aug_state_size = self.pixels * number_of_frames * 2

		self.rows = int(self.table_height)
		self.cols = int(self.table_width)

		# state size = single feature * axes * objects * frames
		self.vec_state_size_sparse = self.rows_sparse * 2 * 2 * number_of_frames

		self.clean_state = np.zeros((2*number_of_frames, self.rows, self.cols))
					
		self.vec_state_size_bio = self.rows * self.cols * number_of_frames *2

	def Convert_bio(self,state):
		batches = state.shape[0]
		D = 1/self.d
		for batch in xrange(batches):
			bio_state = np.copy(self.clean_state)
			for fr in xrange(self.number_of_frames):
				a_idx_x = int((state[batch,self.state_size*fr] + self.table_height) / self.d)
				a_idx_y = int((state[batch,self.state_size*fr+1] + self.table_width/2) / self.d)
				p_idx_x = int((state[batch,self.state_size*fr+4] + self.table_height) / self.d)
				p_idx_y = int((state[batch,self.state_size*fr+5] + self.table_width/2) / self.d)

				a_new_x = int(a_idx_x / D)
				a_new_y = int(a_idx_y / D)

				a_alpha_x = (a_idx_x - (a_new_x+1)*D + 1) / (1-D)
				a_alpha_y = (a_idx_y - (a_new_y+1)*D + 1) / (1-D)
				
				p_new_x = int(a_idx_x / D)
				p_new_y = int(a_idx_y / D)

				p_alpha_x = (p_idx_x - (p_new_x+1)*D + 1) / (1-D)
				p_alpha_y = (p_idx_y - (p_new_y+1)*D + 1) / (1-D)

				bio_state[self.number_of_frames*fr,a_new_x,a_new_y] = a_alpha_x*a_alpha_y
				bio_state[self.number_of_frames*fr,a_new_x+1,a_new_y] = (1-a_alpha_x)*a_alpha_y
				bio_state[self.number_of_frames*fr,a_new_x,a_new_y+1] = a_alpha_x*(1-a_alpha_y)
				bio_state[self.number_of_frames*fr,a_new_x+1,a_new_y+1] = (1-a_alpha_x)*(1-a_alpha_y)
				
				bio_state[self.number_of_frames*fr+1,p_new_x,p_new_y] = p_alpha_x*p_alpha_y
				bio_state[self.number_of_frames*fr+1,p_new_x+1,p_new_y] = (1-p_alpha_x)*p_alpha_y
				bio_state[self.number_of_frames*fr+1,p_new_x,p_new_y+1] = p_alpha_x*(1-p_alpha_y)
				bio_state[self.number_of_frames*fr+1,p_new_x+1,p_new_y+1] = (1-p_alpha_x)*(1-p_alpha_y)
				
			bio_state = bio_state.reshape(1,self.vec_state_size_bio)

			if batch == 0:
				batch_state = bio_state
			else:
				batch_state = np.concatenate((batch_state, bio_state),axis=0)
		return batch_state


	def Convert_sparse(self, state):
		batches = state.shape[0]
		for batch in xrange(batches):
			im_idx = []
			for fr in xrange(self.number_of_frames):
				aX = state[batch,self.state_size*fr]
				agentXf = (aX + self.table_height) / self.d
				a_idx_x = round(agentXf)

				aY = state[batch,self.state_size*fr + 1]
				agentYf = (aY+self.table_width/2) / self.d
				a_idx_y = round(agentYf)

				pX = state[batch,self.state_size*fr + 4]
				puckXf = (pX + self.table_height) / self.d
				p_idx_x = round(puckXf)

				pY = state[batch,self.state_size*fr + 5]
				puckYf = (pY + self.table_width/2) / self.d
				p_idx_y = round(puckYf)

				im_idx.append(int(a_idx_x))
				im_idx.append(int(a_idx_y))
				im_idx.append(int(p_idx_x))
				im_idx.append(int(p_idx_y))

			sparse_state = self.imdb[im_idx,:].reshape(1,self.vec_state_size_sparse)
			
			if batch == 0:
				batch_state = sparse_state
			else:
				batch_state = np.concatenate((batch_state,sparse_state),axis=0)
		return batch_state


	def LoadIMDB(self, filename):
		self.imdb = np.load(filename)

		


