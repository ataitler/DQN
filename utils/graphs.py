from Logger import Logger
import matplotlib.pyplot as plt
import numpy as np

LEN = 100

# libraries
BASE = '/home/ayal/Documents/gym/Code/'
DDPG2_DIR = 'DDPG2/results/logs'
DDPG1_DIR = 'DDPG2/results2/logs'
DQN_DIR = 'DQN_hockey/hockey_DDQN_deepmind/hockey_DQN_5000_V5/logs'
GDQN1_DIR = 'DQN_hockey/hockey_numeric_3points/hockey_multinet1_decay_rate1.2_2_V5/logs'
GDQN2_DIR = 'DQN_hockey/hockey_numeric_3points/hockey_multinet1_decay_rate1.2_2_V6/logs'
GDQN3_DIR = 'DQN_hockey/hockey_numeric_3points/hockey_multinet1_decay_rate1.2_2_V7/logs'
GDQN4_DIR = 'DQN_hockey/hockey_numeric_3points/hockey_multinet1_decay_rate1.2_2_V2/logs'

# loggers
L_DDPG1 = Logger()
L_DDPG1.Load(BASE+DDPG1_DIR)
L_DDPG2 = Logger()
L_DDPG2.Load(BASE+DDPG2_DIR)
L_DQN = Logger()
L_DQN.Load(BASE+DQN_DIR)
L_GDQN1 = Logger()
is_empty = L_GDQN1.Load(BASE+GDQN1_DIR)
L_GDQN2 = Logger()
is_empty = L_GDQN2.Load(BASE+GDQN2_DIR)
L_GDQN3 = Logger()
is_empty = L_GDQN3.Load(BASE+GDQN3_DIR)
L_GDQN4 = Logger()
is_empty = L_GDQN4.Load(BASE+GDQN4_DIR)

j = 1
plt.figure(1)
# Network left
logs = ['network_left', 'network_middle', 'network_right', 'network_random', 'estimated_value']
#logs = ['network_random']
for logname  in logs:
	GDQN1_m = L_GDQN1.GetLogByName(logname)
	GDQN2_m = L_GDQN2.GetLogByName(logname)
	GDQN3_m = L_GDQN3.GetLogByName(logname)
	GDQN4_m = L_GDQN4.GetLogByName(logname)
	DQN_m = L_DQN.GetLogByName(logname)
	DDPG1_m = L_DDPG1.GetLogByName(logname)
	DDPG2_m = L_DDPG2.GetLogByName(logname)
	M1 = len(GDQN1_m)
	M2 = len(DQN_m)
	M3 = len(DDPG1_m)
	m11_avg = []
	m11_up = []
	m11_down = []
	m12_avg = []
	m12_up = []
	m12_down = []
	m13_avg = []
	m13_up = []
	m13_down = []
	m14_avg = []
	m14_up = []
	m14_down = []
	m2_avg = []
	m2_up = []
	m2_down = []
	m3_avg = []
	m3_up = []
	m3_down = []
	mr_avg = []
	mr_up = []
	mr_down = []
	for i in xrange(M1-LEN):
		avg11 = sum(GDQN1_m[i:i+LEN])/LEN*1.0
		var11 = np.sqrt(np.var(GDQN1_m[i:i+LEN]))
		m11_avg.append(avg11)
		m11_up.append(avg11 + var11)
		m11_down.append(avg11 - var11)
		avg12 = sum(GDQN2_m[i:i+LEN])/LEN*1.0
		var12 = np.sqrt(np.var(GDQN2_m[i:i+LEN]))
		m12_avg.append(avg12)
		m12_up.append(avg12 + var12)
		m12_down.append(avg12 - var12)
		avg13 = sum(GDQN3_m[i:i+LEN])/LEN*1.0
		var13 = np.sqrt(np.var(GDQN3_m[i:i+LEN]))
		m13_avg.append(avg13)
		m13_up.append(avg13 + var13)
		m13_down.append(avg13 - var13)
		avg14 = sum(GDQN4_m[i:i+LEN])/LEN*1.0
		var14 = np.sqrt(np.var(GDQN4_m[i:i+LEN]))
		m14_avg.append(avg14)
		m14_up.append(avg14 + var14)
		m14_down.append(avg14 - var14)
		if logname is 'network_random':
			dd = [avg11, avg12, avg13, avg14]
			avg = sum(dd)/len(dd)*1.0
			mr_avg.append(avg)
			varmr = np.sqrt(np.var(dd))
			mr_up.append(avg + varmr)
			mr_down.append(avg - varmr)

	for i in xrange(M2-LEN):
		avg2 = sum(DQN_m[i:i+LEN])/LEN*1.0
		var2 = np.sqrt(np.var(DQN_m[i:i+LEN]))
		m2_avg.append(avg2)
		m2_up.append(avg2 + var2)
		m2_down.append(avg2 - var2)
	
	for i in xrange(M3-LEN):
		avg3 = sum(DDPG1_m[i:i+LEN])/LEN*1.0
		var3 = np.sqrt(np.var(DDPG1_m[i:i+LEN]))
		m3_avg.append(avg3)
		m3_up.append(avg3 + var3)
		m3_down.append(avg3 - var3)


	t1 = np.arange(1,len(m11_avg)+1)
	t2 = np.arange(1,len(m2_avg)+1)
	t3 = np.arange(1,len(m3_avg)+1)
	m11_down = np.array(m11_down)
	m11_up = np.array(m11_up)
	m12_down = np.array(m12_down)
	m12_up = np.array(m12_up)
	m13_down = np.array(m13_down)
	m13_up = np.array(m13_up)
	m14_down = np.array(m14_down)
	m14_up = np.array(m14_up)
	m2_down = np.array(m2_down)
	m2_up = np.array(m2_up)
	m3_down = np.array(m3_down)
	m3_up = np.array(m3_up)

	sp = plt.subplot(2,3,j)
	j += 1
	plt.fill_between(t1, m11_down, m11_up, facecolor = 'blue', linewidth=0.0, alpha = 0.5)
	avg1, = plt.plot(t1, m11_avg,'b')
	plt.fill_between(t1, m12_down, m12_up, facecolor = 'cyan', linewidth=0.0, alpha = 0.5)
	avg2, =	plt.plot(t1, m12_avg,'c')
	plt.fill_between(t1, m13_down, m13_up, facecolor = 'yellow', linewidth=0.0, alpha = 0.5)
	avg3, =	plt.plot(t1, m13_avg,'y')
	plt.fill_between(t1, m14_down, m14_up, facecolor = 'magenta', linewidth=0.0, alpha = 0.5)
	avg4, =	plt.plot(t1, m14_avg,'m')
	plt.fill_between(t2, m2_down, m2_up, facecolor = 'green', linewidth=0.0, alpha = 0.5)
	avg5, =	plt.plot(t2, m2_avg,'g')
	plt.fill_between(t3, m3_down, m3_up, facecolor = 'black', linewidth=0.0, alpha = 0.5)
	avg6, =	plt.plot(t3, m3_avg,'k')
	#sp.legend([avg1, avg2, avg3, avg4, avg5, avg6], ['gDQN1','gDQN2','gDQN3','gDQN4', 'DDQN', 'DDPG'], loc=4)
	
	plt.title(logname)
	plt.xlabel('episodes')
	if logname is not 'estimated_value':
		plt.ylabel('reward')
	else:
		plt.ylabel('average value')

	if logname is 'network_random':
		t = np.arange(1,len(mr_avg)+1)
		mr_up = np.array(mr_up)
		mr_down = np.array(mr_down)
		print t.shape, mr_up.shape, mr_down.shape
		sp = plt.subplot(2,3,6)
		avgr, = plt.plot(t, mr_avg, 'b')
		plt.fill_between(t, mr_down, mr_up, facecolor ='blue', linewidth=0.0, alpha = 0.5)
		plt.title('random seed')
		plt.xlabel('episode')
		plt.ylabel('reward')
		sp.legend([avgr], ['gDQN'], loc=4)

plt.show()

 
