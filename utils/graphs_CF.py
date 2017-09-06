from Logger import Logger
import matplotlib.pyplot as plt
import numpy as np

LEN = 100

# libraries
BASE = '/home/ayal/Documents/gym/Code/DQN_hockey/hockey_numeric_3points/'
DROP_DIR = 'hockey_CF1_drop/logs'
DROPDQN_DIR = 'hockey_CF1_pureDQN2/logs'
#WORB_DIR = 'hockey_CF1_pureDQN/logs'
WORB_DIR = 'hockey_CF1_noupdateRB/logs'
WRB_DIR = 'hockey_CF1_updateRB/logs'

# loggers
L_DROP = Logger()
L_DROP.Load(BASE+DROP_DIR)

L_DROPDQN = Logger()
L_DROPDQN.Load(BASE+DROPDQN_DIR)

L_WORB = Logger()
L_WORB.Load(BASE+WORB_DIR)

L_WRB = Logger()
L_WRB.Load(BASE+WRB_DIR)

j = 1
plt.figure(1)
logs = ['network_left', 'network_middle', 'network_right', 'network_random', 'estimated_value','error']
for logname  in logs:
	DROP_m = L_DROP.GetLogByName(logname)
	DROPDQN_m = L_DROPDQN.GetLogByName(logname)
	WORB_m = L_WORB.GetLogByName(logname)
	WRB_m = L_WRB.GetLogByName(logname)

	M1 = len(DROP_m)
	M2 = len(WORB_m)
	M3 = len(WRB_m)
	M4 = len(DROPDQN_m)

	m1_avg = []
	m1_up = []
	m1_down = []
	m2_avg = []
	m2_up = []
	m2_down = []
	m3_avg = []
	m3_up = []
	m3_down = []
	m4_avg = []
	m4_up = []
	m4_down = []


	for i in xrange(M1-LEN):
		avg = sum(DROP_m[i:i+LEN])/LEN*1.0
		var = np.sqrt(np.var(DROP_m[i:i+LEN]))
		m1_avg.append(avg)
		m1_up.append(avg + var)
		m1_down.append(avg - var)


	for i in xrange(M2-LEN):
		avg = sum(WORB_m[i:i+LEN])/LEN*1.0
		var = np.sqrt(np.var(WORB_m[i:i+LEN]))
		m2_avg.append(avg)
		m2_up.append(avg + var)
		m2_down.append(avg - var)


	for i in xrange(M3-LEN):
		avg = sum(WRB_m[i:i+LEN])/LEN*1.0
		var = np.sqrt(np.var(WRB_m[i:i+LEN]))
		m3_avg.append(avg)
		m3_up.append(avg + var)
		m3_down.append(avg - var)
	
	for i in xrange(M4-LEN):
		avg = sum(DROPDQN_m[i:i+LEN])/LEN*1.0
		var = np.sqrt(np.var(DROPDQN_m[i:i+LEN]))
		m4_avg.append(avg)
		m4_up.append(avg + var)
		m4_down.append(avg - var)

	t1 = np.arange(1,len(m1_avg)+1)
	t2 = np.arange(1,len(m2_avg)+1)
	t3 = np.arange(1,len(m3_avg)+1)
	t4 = np.arange(1,len(m4_avg)+1)
	m1_down = np.array(m1_down)
	m1_up = np.array(m1_up)
	m2_down = np.array(m2_down)
	m2_up = np.array(m2_up)
	m3_down = np.array(m3_down)
	m3_up = np.array(m3_up)
	m4_down = np.array(m4_down)
	m4_up = np.array(m4_up)

	sp = plt.subplot(2,3,j)
	j += 1
	###plt.fill_between(t1, m1_down, m1_up, facecolor = 'black', linewidth=0.0, alpha = 0.5)
	###avg1, = plt.plot(t1, m1_avg,'k')
	plt.fill_between(t2, m2_down, m2_up, facecolor = 'blue', linewidth=0.0, alpha = 0.5)
	avg2, =	plt.plot(t2, m2_avg,'b')
	plt.fill_between(t3, m3_down, m3_up, facecolor = 'red', linewidth=0.0, alpha = 0.5)
	avg3, =	plt.plot(t3, m3_avg,'r')
	plt.fill_between(t4, m4_down, m4_up, facecolor = 'green', linewidth=0.0, alpha = 0.5)
	avg4, =	plt.plot(t4, m4_avg,'g')
	###sp.legend([avg1, avg2, avg3, avg4], ['Normal','dynamic RB ','static RB','Normal DQN'], loc=1)
	sp.legend([avg2, avg3, avg4], ['Static RB ','Dynamic RB','DQN'], loc=1)
	
	#plt.title(logname)
	plt.xlabel('episodes')
	if logname is not 'estimated_value':
		plt.ylabel('reward')
	elif logname is not 'error':
		plt.ylabel('average value')
	else:
		plt.ylabel('MSE')
	
	if logname is 'estimated_value':
		plt.title('Estimated value')
	elif logname is 'network_left':
		plt.title('Puck left')
	elif logname is 'network_middle':
		plt.title('Puck middle')
	elif logname is 'network_right':
		plt.title('Puck right')
	elif logname is 'network_random':
		plt.title('Puck random')
	else:
		plt.title(logname)




	#if logname is 'network_random':
	#	t = np.arange(1,len(mr_avg)+1)
	#	mr_up = np.array(mr_up)
	#	mr_down = np.array(mr_down)
	#	print t.shape, mr_up.shape, mr_down.shape
	#	sp = plt.subplot(2,3,6)
	#	avgr, = plt.plot(t, mr_avg, 'b')
	#	plt.fill_between(t, mr_down, mr_up, facecolor ='blue', linewidth=0.0, alpha = 0.5)
	#	plt.title('random seed')
	#	plt.xlabel('episode')
	#	plt.ylabel('reward')
	#	sp.legend([avgr], ['gDQN'], loc=4)

plt.show()

 
