import numpy as np
from collections import deque
import bisect
import os

###############################################
#	Replay Buffer
###############################################
class ReplayBuffer():
	populated = 0
	def __init__(self, state_space, action_space, buffer_size):
		self.buffer_size = buffer_size
		self.state_size = state_space
		self.action_size = action_space
		self.bucket = TransitionBucket(self.buffer_size)

	def SampleMiniBatch(self, batch):
		S, A, R, Stag, D, top = self.bucket.Get(batch)
		R = np.array(R).reshape(top,1)
		A = np.array(A).reshape(top,1)
		S = np.array(S).reshape(top,self.state_size)
		Stag = np.array(Stag).reshape(top,self.state_size)
		D = np.array(D).reshape(top,1)
		return S, A, R, Stag, D, top

	def GetMiniBatch(self,size, iteration):
		S, A, R, Stag, D, top = self.bucket.GetByIndex(size, iteration)
		if top > 0:
			R = np.array(R).reshape(top,1)
			A = np.array(A).reshape(top,1)
			S = np.array(S).reshape(top,self.state_size)
			Stag = np.array(Stag).reshape(top,self.state_size)
			D = np.array(D).reshape(top,1)
			return S, A, R, Stag, D, top
		else:
			return np.nan, np.nan, np.nan, np.nan, np.nan, top

	def StoreTransition(self, s_t, a_t, r_t, s_t_next, d_t=0):
		s_t = s_t.reshape(1, self.state_size)
		s_t_next = s_t_next.reshape(1, self.state_size)
		a_t = a_t.reshape(1, self.action_size)
		r_t = r_t.reshape(1, 1)
		d_t = np.array([d_t]).reshape(1, 1)
		Transition = [s_t, a_t, r_t, s_t_next, d_t]
		self.bucket.Add(Transition, 0)

	def SaveBuffer(self, save_file):
		S,A,R,Stag,D,pop = self.bucket.GetBucket()
		np.savez_compressed(save_file, S=np.array(S), A=np.array(A), R=np.array(R), Stag=np.array(Stag), D=np.array(D), \
		pop=pop, bs=self.buffer_size, ss=self.state_size, acs=self.action_size)

	def LoadBuffer(self, save_file):
		if not os.path.exists(save_file+'.npz'):
			return False
		arrays = np.load(save_file+'.npz')
		S = deque(arrays['S'])
		A = deque(arrays['A'])
		R = deque(arrays['R'])
		Stag = deque(arrays['Stag'])
		D = deque(arrays['D'])
		populated = arrays['pop'].reshape(1)[0]
		self.bucket.SetBucket(S,A,R,Stag,D,populated)
		self.buffer_size = arrays['bs'].reshape(1)[0]
		self.state_space = arrays['ss'].reshape(1)[0]
		self.action_space = arrays['acs'].reshape(1)[0]
		return True
		
	def EmptyBuffer(self):
		self.bucket.Empty()
	
	def GetOccupency(self):
		return self.bucket.GetOccupency()


###############################################
#	Replay Buffer Sampler
###############################################
class ReplayBufferSampler():
	populated = 0
	def __init__(self, state_space, action_space, buffer_size):
		self.buffer_size = buffer_size
		self.state_size = state_space
		self.action_size = action_space
		self.bucket = TransitionBucket(self.buffer_size)

	def SampleMiniBatch(self, batch):
		S, A, R, Stag, D, top = self.bucket.Get(batch)
		R = np.array(R).reshape(top,1)
		A = np.array(A).reshape(top,1)
		S = np.array(S).reshape(top,self.state_size)
		Stag = np.array(Stag).reshape(top,self.state_size)
		D = np.array(D).reshape(top,1)
		return S, A, R, Stag, D, top

	def StoreTransition(self, s_t, a_t, r_t, s_t_next, d_t=0):
		s_t = s_t.reshape(1, self.state_size)
		s_t_next = s_t_next.reshape(1, self.state_size)
		a_t = a_t.reshape(1, self.action_size)
		r_t = r_t.reshape(1, 1)
		d_t = np.array([d_t]).reshape(1, 1)
		Transition = [s_t, a_t, r_t, s_t_next, d_t]
		self.bucket.Add(Transition, 0)

	def SaveBuffer(self, save_file):
		S,A,R,Stag,D,pop = self.bucket.GetBucket()
		np.savez_compressed(save_file, S=np.array(S), A=np.array(A), R=np.array(R), Stag=np.array(Stag), D=np.array(D), \
		pop=pop, bs=self.buffer_size, ss=self.state_size, acs=self.action_size)

	def LoadBuffer(self, save_file):
		if not os.path.exists(save_file+'.npz'):
			return False
		arrays = np.load(save_file+'.npz')
		S = deque(arrays['S'])
		A = deque(arrays['A'])
		R = deque(arrays['R'])
		Stag = deque(arrays['Stag'])
		D = deque(arrays['D'])
		populated = arrays['pop'].reshape(1)[0]
		self.bucket.SetBucket(S,A,R,Stag,D,populated)
		self.buffer_size = arrays['bs'].reshape(1)[0]
		self.state_space = arrays['ss'].reshape(1)[0]
		self.action_space = arrays['acs'].reshape(1)[0]
		return True

	def SampleBuffer(self, size, save_file):
		current_size = self.bucket.GetOccupency()
		if size > current_size:
			print "please specify size not larger than current buffer file: ", current_size
			return
		S, A, R, Stag, D, pop = self.bucket.GetBucket()
		prob = 1.0*size/pop
		a = np.random.uniform(0,1,pop)
		a = np.round(np.clip(a + 0.5 - prob, 0, 1, a)).astype(int)
		inds = []
		for i in xrange(0,pop):
			if a[i] == 0:
				inds.append(i)
		#print inds
		sampled_size = pop-a.sum()
		Rt = deque()
		St = deque()
		At = deque()
		Stagt = deque()
		Dt = deque()
			
		for i in inds:
			Rt.append(R[i])
			At.append(A[i])
			St.append(S[i])
			Stagt.append(Stag[i])
			Dt.append(D[i])
		sampled_S = np.array(St).reshape(sampled_size, self.state_size)
		sampled_A = np.array(At).reshape(sampled_size, self.action_size)
		sampled_R = np.array(Rt).reshape(sampled_size, 1)
		sampled_Stag = np.array(Stagt).reshape(sampled_size, self.state_size)
		sampled_D = np.array(Dt).reshape(sampled_size, 1)
	
		size = len(inds)
		np.savez_compressed(save_file, S=np.array(sampled_S), A=np.array(sampled_A), R=np.array(sampled_R), Stag=np.array(sampled_Stag), D=np.array(sampled_D), \
				pop=sampled_size, bs=sampled_size, ss=self.state_size, acs=self.action_size)

	def EmptyBuffer(self):
		self.bucket.Empty()

	def GetOccupency(self):
		return self.bucket.GetOccupency()
	

###############################################
#	Bipoloar Replay Buffer
###############################################
class BipolarReplayBuffer():
	def __init__(self, state_space, action_space, buffer_size):
		self.buffer_size = buffer_size
		self.state_size = state_space
		self.action_size = action_space
		self.positiveBucket = TransitionBucket(int(round(self.buffer_size/2)))
		self.negativeBucket = TransitionBucket(int(round(self.buffer_size/2)))

	def SampleMiniBatch(self, batch, rho):
		p_batch = int(round(batch*rho))
		pS, pA, pR, pStag, pD, ptop = self.positiveBucket.Get(p_batch)
		nS, nA, nR, nStag, nD, ntop = self.negativeBucket.Get(batch - p_batch)

		pR.extend(nR)
		pA.extend(nA)
		pS.extend(nS)
		pStag.extend(nStag)
		pD.extend(nD)
		top = ntop+ptop

		R = np.array(pR).reshape(top,1)
		print R
		A = np.array(pA).reshape(top,1)
		S = np.array(pS).reshape(top,self.state_size)
		Stag = np.array(pStag).reshape(top,self.state_size)
		D = np.array(pD).reshape(top,1)
		return S, A, R, Stag, D, top

	def StoreTransition(self, s_t, a_t, r_t, s_t_next, d_t=0):
		s_t = s_t.reshape(1, self.state_size)
		s_t_next = s_t_next.reshape(1, self.state_size)
		a_t = a_t.reshape(1, self.action_size)
		r_t = r_t.reshape(1, 1)
		d_t = np.array([d_t]).reshape(1, 1)
		Transition = [s_t, a_t, r_t, s_t_next, d_t]
		if r_t[0,0] > 0:
			self.positiveBucket.Add(Transition,0)
		else:
			self.negativeBucket.Add(Transition,0)

	def SaveBuffer(self, save_file):
		pS, pA, pR, pStag, pD, ppop = self.positiveBucket.GetBucket()
		nS, nA, nR, nStag, nD, npop = self.negativeBucket.GetBucket()
		np.savez_compressed(save_file, pS=np.array(pS), pA=np.array(pA), pR=np.array(pR), pStag=np.array(pStag), pD=np.array(pD), \
			ppop=ppop, nS=np.array(nS), nA=np.array(nA), nR=np.array(nR), nStag=np.array(nStag), nD=np.array(nD), npop=npop, \
			bs=self.buffer_size, ss=self.state_size, acs=self.action_size)

	def LoadBuffer(self, save_file):
		if not os.path.exists(save_file+'.npz'):
			return False
		arrays = np.load(save_file+'.npz')
		pS = deque(arrays['pS'])
		pA = deque(arrays['pA'])
		pR = deque(arrays['pR'])
		pStag = deque(arrays['pStag'])
		pD = deque(arrays['pD'])
		ppopulated = arrays['ppop'].reshape(1)[0]
		self.positiveBucket.SetBucket(pS, pA, pR, pStag, pD, ppopulated)
		nS = deque(arrays['nS'])
		nA = deque(arrays['nA'])
		nR = deque(arrays['nR'])
		nStag = deque(arrays['nStag'])
		nD = deque(arrays['nD'])
		npopulated = arrays['npop'].reshape(1)[0]
		self.negativeBucket.SetBucket(nS, nA, nR, nStag, nD, npopulated)
		self.buffer_size = arrays['bs'].reshape(1)[0]
		self.state_space = arrays['ss'].reshape(1)[0]
		self.action_space = arrays['acs'].reshape(1)[0]
		return True

	def EmptyBuffer(self):
		self.positiveBucket.Empty()
		self.negativeBucket.Empty()
	
	def GetOccupency(self):
		return self.positiveBucket.GetOccupency() + self.negativeBucket.GetOccupency()


###############################################
#	TripolarReplay Buffer
###############################################
class TripolarReplayBuffer():
	def __init__(self, state_space, action_space, buffer_size, sorted_buffer = False):
		self.sorted_buffer = sorted_buffer
		self.buffer_size = buffer_size
		self.state_size = state_space
		self.action_size = action_space
		self.positive_size = int(round(self.buffer_size*1/20))
		self.negative_size = int(round(self.buffer_size*1/20))
	  	self.sparse_size = int(round(self.buffer_size*18/20))
		if self.sorted_buffer:
			self.positiveBucket = SortedTransitionBucket(self.positive_size)
		else:
			self.positiveBucket = TransitionBucket(self.positive_size)
		self.sparseBucket = TransitionBucket(self.sparse_size)
		self.negativeBucket = TransitionBucket(self.negative_size)
		self.positiveSize = 0
		self.sparseSize = 0
		self.negativeSize = 0

	def SampleMiniBatch(self, batch, rho_p, rho_n):
		if rho_p + rho_n > 1:
			rho_p = self.positiveSize*1.0/((self.positiveSize+self.negativeSize+self.sparseSize)*1.0)
			rho_n = self.negativeSize*1.0/((self.positiveSize+self.negativeSize+self.sparseSize)*1.0)
		ratio_p = batch*rho_p
		i,d = divmod(ratio_p,1)
		if d > np.random.uniform():
			p_batch = int(np.ceil(ratio_p))
		else:
			p_batch = int(np.floor(ratio_p))

		ratio_n = batch*rho_n
		i,d = divmod(ratio_n,1)
		if d > np.random.uniform():
			n_batch = int(np.ceil(ratio_n))
		else:
			n_batch = int(np.floor(ratio_n))

		pS, pA, pR, pStag, pD, ptop = self.positiveBucket.Get(p_batch)
		nS, nA, nR, nStag, nD, ntop = self.negativeBucket.Get(n_batch)
		S, A, R, Stag, D, stop = self.sparseBucket.Get(np.maximum(0,batch - n_batch - p_batch))
		R.extend(nR)
		R.extend(pR)
		A.extend(nA)
		A.extend(pA)
		S.extend(nS)
		S.extend(pS)
		Stag.extend(nStag)
		Stag.extend(pStag)
		D.extend(nD)
		D.extend(pD)
		top = stop + ntop + ptop

		R = np.array(R).reshape(top,1)
		A = np.array(A).reshape(top,1)
		S = np.array(S).reshape(top,self.state_size)
		Stag = np.array(Stag).reshape(top,self.state_size)
		D = np.array(D).reshape(top,1)
		return S, A, R, Stag, D, top

	def StoreTransition(self, s_t, a_t, r_t, s_t_next, d_t=0):
		s_t = s_t.reshape(1, self.state_size)
		s_t_next = s_t_next.reshape(1, self.state_size)
		a_t = a_t.reshape(1, self.action_size)
		r_t = r_t.reshape(1, 1)
		d_t = np.array([d_t]).reshape(1, 1)
		Transition = [s_t, a_t, r_t, s_t_next, d_t]
		if r_t[0,0] > 0:
			self.positiveSize = self.positiveBucket.Add(Transition,0)
		elif r_t[0,0] < -10:
			self.negativeSize = self.negativeBucket.Add(Transition,0)
		else:
			self.sparseSize = self.sparseBucket.Add(Transition,0)

	def SaveBuffer(self, save_file):
		nS, nA, nR, nStag, nD, npop = self.negativeBucket.GetBucket()
		sS, sA, sR, sStag, sD, spop = self.sparseBucket.GetBucket()
		if self.sorted_buffer:
			pRank, pS, pA, pR, pStag, pD, ppop = self.positiveBucket.GetBucket()
			np.savez_compressed(save_file, pRank=pRank, pS=np.array(pS), pA=np.array(pA), pR=np.array(pR), pStag=np.array(pStag), pD=np.array(pD), \
				ppop=ppop, nS=np.array(nS), nA=np.array(nA), nR=np.array(nR), nStag=np.array(nStag), nD=np.array(nD), npop=npop, \
				sS=np.array(sS), sA=np.array(sA), sR=np.array(sR), sStag=np.array(sStag), sD=np.array(sD), spop=spop, \
				bs=self.buffer_size, ss=self.state_size, acs=self.action_size, sb=np.array([self.sorted_buffer]))
		else:
			pS, pA, pR, pStag, pD, ppop = self.positiveBucket.GetBucket()
			np.savez_compressed(save_file, pS=np.array(pS), pA=np.array(pA), pR=np.array(pR), pStag=np.array(pStag), pD=np.array(pD), \
				ppop=ppop, nS=np.array(nS), nA=np.array(nA), nR=np.array(nR), nStag=np.array(nStag), nD=np.array(nD), npop=npop, \
				sS=np.array(sS), sA=np.array(sA), sR=np.array(sR), sStag=np.array(sStag), sD=np.array(sD), spop=spop, \
				bs=self.buffer_size, ss=self.state_size, acs=self.action_size, sb=np.array([self.sorted_buffer]))
	
	def LoadBuffer(self, save_file):
		if not os.path.exists(save_file+'.npz'):
			return False
		arrays = np.load(save_file+'.npz')
		self.sorted_buffer = arrays['sb'][0]
		if self.sorted_buffer:
			pS = arrays['pS']
			pA = arrays['pA']
			pR = arrays['pR']
			pStag = arrays['pStag']
			pD = arrays['pD']
			self.positiveSize = arrays['ppop'].reshape(1)[0]
			pRank = arrays['pRank']
			self.positiveBucket.SetBucket(pRank, pS, pA, pR, pStag, pD, self.positiveSize)
		else:
			pS = deque(arrays['pS'])
			pA = deque(arrays['pA'])
			pR = deque(arrays['pR'])
			pStag = deque(arrays['pStag'])
			pD = deque(arrays['pD'])
			self.positiveSize = arrays['ppop'].reshape(1)[0]
			self.positiveBucket.SetBucket(pS, pA, pR, pStag, pD, self.positiveSize)

		nS = deque(arrays['nS'])
		nA = deque(arrays['nA'])
		nR = deque(arrays['nR'])
		nStag = deque(arrays['nStag'])
		nD = deque(arrays['nD'])
		self.negativeSize = arrays['npop'].reshape(1)[0]
		self.negativeBucket.SetBucket(nS, nA, nR, nStag, nD, self.negativeSize)
		
		sS = deque(arrays['sS'])
		sA = deque(arrays['sA'])
		sR = deque(arrays['sR'])
		sStag = deque(arrays['sStag'])
		sD = deque(arrays['sD'])
		self.sparseSize = arrays['spop'].reshape(1)[0]
		self.sparseBucket.SetBucket(sS, sA, sR, sStag, sD, self.sparseSize)

		self.buffer_size = arrays['bs'].reshape(1)[0]
		self.state_space = arrays['ss'].reshape(1)[0]
		self.action_space = arrays['acs'].reshape(1)[0]
		return True

	def EmptyBuffer(self):
		self.positiveBucket.Empty()
		self.negativeBucket.Empty()
		self.sparseBucket.Empty()
	
	def GetOccupency(self):
		return self.positiveBucket.GetOccupency() + self.negativeBucket.GetOccupency() + self.sparseBucket.GetOccupency()

	def PrintOccupency(self):
		print self.positiveBucket.GetOccupency(),"/",self.positive_size, " ", \
			self.negativeBucket.GetOccupency(),"/",self.negative_size, " ", \
			self.sparseBucket.GetOccupency(),"/",self.sparse_size


###############################################
#	Trajecotry Replay Buffer
###############################################
class TrajectoryReplayBuffer():
	def __init__(self, state_space, action_space, buffer_size, max_episode_length):
		self.max_len = max_episode_length
		self.state_space = state_space
		self.action_space = action_space
		self.buffer_size = buffer_size
		self.current_trajectory = -1

		self._initContainers()
		self.bucket = TrajectoryBucket(self.buffer_size)
	
	def AddTransition(self, s_t, a_t, r_t, s_tag, d_t):
		if self.TransitionCounter > self.max_len:
			self.FinilizeTrajectory()

		self.St.append(s_t.reshape(self.state_space))
		self.Rt.append(r_t)
		self.At.append(a_t)
		self.Stag.append(s_tag.reshape(self.state_space))
		self.Dt.append(d_t)
		self.TransitionCounter += 1

	def FinilizeTrajectory(self):
		trajectory = [np.array(self.St), np.array(self.At), np.array(self.Rt), np.array(self.Stag), np.array(self.Dt)]
		self.bucket.Add(trajectory,0)
		self._initContainers()
		self.current_trajectory += 1	

	def SampleTrajectory(self):
		return self.bucket.Get()

	def GetLastTrajectory(self):
		return self.bucket.GetIndex(self.current_trajectory % self.buffer_size)

	def SaveBuffer(self, save_file):
		s1, p1 = self.bucket.GetBucket()
		np.savez_compressed(save_file, S=s1, pop=np.array([p1]),bs=np.array([self.buffer_size]), \
			ss=np.array([self.state_space]), acs=np.array([self.action_space]), maxlen=np.array([self.max_len]), \
			cur=np.array([self.current_trajectory % self.buffer_size]))
		self._initContainers()
	
	def LoadBuffer(self, save_file):
		if not os.path.exists(save_file+'.npz'):
			return False
		temp = np.load(save_file+".npz")
		self.buffer_size = temp['bs'].reshape(1)[0]
		self.bucket.SetBucket(deque(temp['S']), temp['pop'].reshape(1)[0], self.buffer_size)
		self.state_space = temp['ss'].reshape(1)[0]
		self.action_space = temp['acs'].reshape(1)[0]
		self.populated = temp['pop'].reshape(1)[0]
		self.max_len = temp['maxlen'].reshape(1)[0]
		self.current_trajectory = temp['cur'].reshape(1)[0]
		return True

	def EmptyBuffer(self):
		self._initContainers()
		self.Buffer = deque(maxlen = self.buffer_size)
		self.populated = 0
	
	def GetOccupency(self):
		return self.bucket.GetOccupency()

	def GetR(self):
		return self.bucket.GetBucket()[0]

	def GetRasArray(self):
		return np.array(self.bucket.GetBucket()[0])

	def _initContainers(self):
		self.Rt = deque(maxlen = self.max_len)
		self.St = deque(maxlen = self.max_len)
		self.At = deque(maxlen = self.max_len)
		self.Stag = deque(maxlen = self.max_len)
		self.Dt = deque(maxlen = self.max_len)
		self.TransitionCounter = 0


###############################################
#	Bipolar Trajectory Replay Buffer
###############################################
class BipolarTrajectoryReplayBuffer():
	def __init__(self, state_space, action_space, buffer_size, max_episode_length):
		self.max_len = max_episode_length
		self.state_space = state_space
		self.action_space = action_space
		self.buffer_size = buffer_size

		self._initContainers()
		self.positiveBucket = TrajectoryBucket(int(round(self.buffer_size/2)))
		self.negativeBucket = TrajectoryBucket(int(round(self.buffer_size/2)))
	
	def AddTransition(self, s_t, a_t, r_t, s_tag, d_t):
		if self.TransitionCounter > self.max_len:
			self.FinilizeTrajectory()

		self.St.append(s_t.reshape(self.state_space))
		self.Rt.append(r_t)
		self.At.append(a_t)
		self.Stag.append(s_tag.reshape(self.state_space))
		self.Dt.append(d_t)
		self.totalR += r_t
		self.TransitionCounter += 1

	def FinilizeTrajectory(self):
		trajectory = [np.array(self.St), np.array(self.At), np.array(self.Rt), np.array(self.Stag), np.array(self.Dt)]
		if self.totalR > 0:
			self.positiveBucket.Add(trajectory, self.totalR)
		else:
			self.negativeBucket.Add(trajectory, self.totalR)
		self._initContainers()	

	def SampleTrajectory(self, rho):
		if rho > np.random.uniform():
			if self.positiveBucket.GetOccupency() > 0:
				return self.positiveBucket.Get()
			else:
				return self.negativeBucket.Get()

		else:
			if self.negativeBucket.GetOccupency() > 0:
				return self.negativeBucket.Get()
			else:
				return self.positiveBucket.Get()

	def SaveBuffer(self, save_file):
		s1, p1 = self.positiveBucket.GetBucket()
		s2, p2 = self.negativeBucket.GetBucket()
		np.savez_compressed(save_file, S1=s1, pop1=np.array([p1]), S2=s2, pop2=np.array([p2]), bs=np.array([self.buffer_size]), \
			ss=np.array([self.state_space]), acs=np.array([self.action_space]), maxlen=np.array([self.max_len]))
		self._initContainers()
	
	def LoadBuffer(self, save_file):
		if not os.path.exists(save_file+'.npz'):
			return False
		temp = np.load(save_file+".npz")
		self.buffer_size = temp['bs'].reshape(1)[0]

		S1 = deque(temp['S1'])
		p1 = temp['pop1'].reshape(1)[0]
		self.positiveBucket.SetBucket(S1, p1, int(round(self.buffer_size/2)))
		S2 = deque(temp['S2'])
		p2 = temp['pop2'].reshape(1)[0]
		self.negativeBucket.SetBucket(S2, p2, int(round(self.buffer_size/2)))
		
		self.state_space = temp['ss'].reshape(1)[0]
		self.action_space = temp['acs'].reshape(1)[0]
		self.max_len = temp['maxlen'].reshape(1)[0]
		self._initContainers()
		return True

	def EmptyBuffer(self):
		self._initContainers()
		self.positiveBucket.Empty()
		self.negativeBucket.Empty()
	
	def GetOccupency(self):
		return self.positiveBucket.GetOccupency() + self.negativeBucket.GetOccupency()

	def GetR(self):
		return self.positiveBucket.GetBucket()[0], self.negativeBucket.GetBucket()[0]

	def _initContainers(self):
		self.Rt = deque(maxlen = self.max_len)
		self.St = deque(maxlen = self.max_len)
		self.At = deque(maxlen = self.max_len)
		self.Stag = deque(maxlen = self.max_len)
		self.Dt = deque(maxlen = self.max_len)
		self.totalR = 0
		self.TransitionCounter = 0


###############################################
#	Trajectory Bucket
###############################################
class TrajectoryBucket():
	def __init__(self, bucket_size):
		self.bucket_size = bucket_size
		self.bucket = deque(maxlen = self.bucket_size)
		self.populated = 0

	def Add(self, trajectory, rank):
		self.bucket.append(trajectory)
		self.populated += 1

	def Get(self):
		i = np.random.randint(self.populated)
		return self.bucket[i]

	def GetIndex(self, idx):
		return self.bucket[idx]

	def GetBucket(self):
		return self.bucket, self.populated

	def SetBucket(self, bucket, populated, size):
		self.bucket = bucket
		self.populated = populated
		self.bucket_size = size

	def GetOccupency(self):
		return self.populated	
	
	def Empty(self):
		self.populated = 0
		self.bucket = deque(max_len = self.bucket_size)


###############################################
#	Transition Bucket
###############################################
class TransitionBucket():
	def __init__(self, bucket_size):
		self.bucket_size = bucket_size
		self.Empty()

	def Empty(self):
		self.R = deque(maxlen=self.bucket_size)
		self.A = deque(maxlen=self.bucket_size)
		self.S = deque(maxlen=self.bucket_size)
		self.Stag = deque(maxlen=self.bucket_size)
		self.D = deque(maxlen=self.bucket_size)
		self.populated = 0

	def Add(self, Transition, rank):
		# Transition = s_t, a_t, r_t, s_t_next, d_t
		self.populated += 1
		if self.populated > self.bucket_size:
			self.S.popleft()
			self.Stag.popleft()
			self.A.popleft()
			self.R.popleft()
			self.D.popleft()
			self.populated = self.bucket_size

		self.S.append(Transition[0])
		self.Stag.append(Transition[3])
		self.A.append(Transition[1])
		self.R.append(Transition[2])
		self.D.append(Transition[4])

		return self.populated

	def Get(self, batch):
		top = min(batch, self.populated)
		batch = np.random.randint(self.populated, size=top)
		
		Rt = deque()
		St = deque()
		At = deque()
		Stag = deque()
		Dt = deque()
			
		for i in batch:
			Rt.append(self.R[i])
			At.append(self.A[i])
			St.append(self.S[i])
			Stag.append(self.Stag[i])
			Dt.append(self.D[i])
		
		return St, At, Rt, Stag, Dt, top

	def GetByIndex(self, size, iteration):
		top = min(size*(iteration+1), self.populated)
		inds = range(size*iteration, top)
		
		Rt = deque()
		St = deque()
		At = deque()
		Stag = deque()
		Dt = deque()
			
		for i in inds:
			Rt.append(self.R[i])
			At.append(self.A[i])
			St.append(self.S[i])
			Stag.append(self.Stag[i])
			Dt.append(self.D[i])
		
		return St, At, Rt, Stag, Dt, len(inds)

	def GetBucket(self):
		return self.S, self.A, self.R, self.Stag, self.D, self.populated

	def SetBucket(self, S, A, R, Stag, D, populated):
		self.S = S
		self.A = A
		self.R = R
		self.Stag = Stag
		self.D = D
		self.populated = populated

	def GetOccupency(self):
		return self.populated


###############################################
#	Sorted Transition Bucket
###############################################
class SortedTransitionBucket():
	def __init__(self, bucket_size):
		self.bucket_size = bucket_size
		self.Empty()

	def Empty(self):
		self.rank = 0
		self.R = 0
		self.A = 0
		self.S = 0
		self.Stag = 0
		self.D = 0
		self.populated = 0

	def Add(self, Transition, rank):
		# Transition = s_t, a_t, r_t, s_t_next, d_t
		self.populated += 1
		if self.populated == 1:
			self.S = np.array([Transition[0]])
			self.A = np.array([Transition[1]])
			self.R = np.array([Transition[2]])
			self.Stag = np.array([Transition[3]])
			self.D = np.array([Transition[4]])
			self.rank = np.array([1.0/Transition[2][0][0]])
		else:
			if self.populated > self.bucket_size:
				self.populated = self.bucket_size
				self.S = np.delete(self.S,self.populated-1,axis=0)
				self.A = np.delete(self.A,self.populated-1,axis=0)
				self.R = np.delete(self.R,self.populated-1,axis=0)
				self.Stag = np.delete(self.Stag,self.populated-1,axis=0)
				self.D = np.delete(self.D,self.populated-1,axis=0)
				self.rank = np.delete(self.rank, self.populated-1)
			
			i = bisect.bisect(self.rank, Transition[2][0][0])
			self.rank = np.insert(self.rank, i, 1.0/Transition[2][0][0])
			self.S = np.insert(self.S, i, Transition[0],axis=0)
			self.A = np.insert(self.A, i, Transition[1],axis=0)
			self.R = np.insert(self.R, i, Transition[2],axis=0)
			self.Stag = np.insert(self.Stag, i, Transition[3],axis=0)
			self.D = np.insert(self.D, i, Transition[4],axis=0)
		
		return self.populated

	def Get(self, batch):
		top = min(batch, self.populated)
		if top == 0:
			return deque(),deque(),deque(),deque(),deque(), 0		
		batch = np.random.randint(self.populated, size=top)	# unifrom distribution
		#batch = np.clip(np.round(np.random.exponential(np.sqrt(self.populated),top)).astype(int), 0, self.populated-1)
		St = deque(self.S[batch])
		At = deque(self.A[batch])
		Rt = deque(self.R[batch])
		Stag = deque(self.Stag[batch])
		Dt = deque(self.D[batch])
		return St, At, Rt, Stag, Dt, top

	def GetBucket(self):
		return self.rank, self.S, self.A, self.R, self.Stag, self.D, self.populated

	def SetBucket(self, rank, S, A, R, Stag, D, populated):
		self.rank = rank
		self.S = S
		self.A = A
		self.R = R
		self.Stag = Stag
		self.D = D
		self.populated = populated

	def GetOccupency(self):
		return self.populated



#R = TrajectoryReplayBuffer(3, 1, 10, 10)
#R.LoadBuffer('buffer')
#R2 = R.GetRasArray()
#print R2.shape
#for episode in xrange(10):
#	for step in xrange(10):
#		st = np.random.uniform(0,1,3).reshape(1,3)
#		at = np.random.uniform(4,5)
#		rt = 1
#		stag = np.random.uniform(0,1,3).reshape(1,3)
#		tofin = np.random.uniform(0,1)
#		if tofin > 0.8:
#			dt = 1
#		else:
#			dt = 0
#		
#		R.AddTransition(st,np.array([at]),np.array([rt]),stag,dt)
#		
#		if dt == 1:
#			break
#	R.FinilizeTrajectory()
#
#T = R.SampleTrajectory()
#print T[0].shape
#print T[3].shape
#print T[1].shape
#print T[2].shape
#print T[4]
	
#R2 = R.GetRasArray()
#print R2.shape

#R.SaveBuffer("buffer")
