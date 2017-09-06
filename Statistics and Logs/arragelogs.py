from utils.Logger import Logger
import matplotlib.pyplot as plt
import numpy as np

num = 200
OUTLOG = 'statistics/log_gDQN'
Base = '/home/ayal/Documents/gym/Code/'
DQN_home = 'DQN_hockey/hockey_DDQN_deepmind/hockey_DQN_'+str(num)+'_V'
DDPG_home = 'DDPG2/results'
gDQN_home = 'DQN_hockey/hockey_numeric_3points/hockey_multinet1_decay_rate1.2_2_V'

gDQNd = [Base+gDQN_home+'7/logs',Base+gDQN_home+'6/logs',Base+gDQN_home+'5/logs',Base+gDQN_home+'2/logs']#,Base+gDQN_home+'4/logs']
DDPGd = [Base+DDPG_home+'/logs',Base+DDPG_home+'1/logs',Base+DDPG_home+'2/logs',Base+DDPG_home+'3/logs',Base+DDPG_home+'4/logs']
DQNd = [Base+DQN_home+'1/logs',Base+DQN_home+'2/logs',Base+DQN_home+'3/logs',Base+DQN_home+'4/logs',Base+DQN_home+'5/logs']


numlogs = len(gDQNd)
LEN = 100

L = Logger()
L.AddNewLog('network_left')
L.AddNewLog('network_left_up')
L.AddNewLog('network_left_down')
L.AddNewLog('network_middle')
L.AddNewLog('network_middle_up')
L.AddNewLog('network_middle_down')
L.AddNewLog('network_right')
L.AddNewLog('network_right_up')
L.AddNewLog('network_right_down')
L.AddNewLog('network_random')
L.AddNewLog('network_random_up')
L.AddNewLog('network_random_down')
L.AddNewLog('estimated_value')
L.AddNewLog('estimated_value_up')
L.AddNewLog('estimated_value_down')


logs = ['network_left','network_middle','network_right','network_random','estimated_value']
DQNl = []
for i in xrange(numlogs):
	DQNl.append(Logger())
	DQNl[i].Load(gDQNd[i])
	#DQNl[i].Load(DDPGd[i])
	#DQNl[i].Load(DQNd[i])

avg = []
for logname in logs:
	all_logs = []
	for i in xrange(numlogs):
		all_logs.append(DQNl[i].GetLogByName(logname))
	length = len(all_logs[i])

	avg = []
	for i in xrange(length):
		temp = []
		for l in xrange(numlogs):
			temp.append(all_logs[l][i])
		avg.append(sum(temp)/len(temp)*1.0)			

	avgep = []
	var = 0
	avgep_up = []
	avgep_down = []
	for i in xrange(length-LEN):
		a = sum(avg[i:i+LEN])/LEN*1.0
		var = np.sqrt(np.var(avg[i:i+LEN]))
		avgep.append(a)
		avgep_up.append(avgep[i] + var)
		avgep_down.append(avgep[i] - var)
		L.AddRecord(logname,avgep[i])
		L.AddRecord(logname+'_up',avgep_up[i])
		L.AddRecord(logname+'_down',avgep_down[i])
	
L.Save(OUTLOG)
t = np.arange(1,len(avgep)+1)
var_up = np.array(avgep_up)
var_down = np.array(avgep_down)
plt.figure(1)
plt.fill_between(t,var_down, var_up, facecolor='blue', linewidth=0.0,alpha=0.5)
plt.plot(t,avgep,'b')
plt.show()



