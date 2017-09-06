import numpy as np
from utils.Logger import Logger
import matplotlib.pyplot as plt


BASE = '/home/ayal/Documents/gym/Code/DQN_hockey/statistics/'
files = ['log_DDQN200']
names = ['DDQN_1000', 'DDPG', 'gDQN' ]
logs = ['network_left','network_middle', 'network_right', 'network_random', 'estimated_value']
lognames = ['error','Puck left', 'Puck middle', 'Puck right', 'Puck random', 'Estimated value']

readers = []
for i in xrange(len(files)):
	readers.append(Logger())
	fname = BASE + files[i]
	readers[i].Load(BASE + files[i])


for log_index in xrange(len(logs)):
	log = logs[log_index]
	fig = plt.figure(log_index+1)

	p1 = readers[0].GetLogByName(log)
	p1_up = readers[0].GetLogByName(log+'_up')
	p1_down = readers[0].GetLogByName(log+'_down')
	t1 = np.arange(1,len(p1)+1)
	
	#p2 = readers[1].GetLogByName(log)
	#p2_up = readers[1].GetLogByName(log+'_up')
	#p2_down = readers[1].GetLogByName(log+'_down')
	#t2 = np.arange(1,len(p2)+1)

	#p3 = readers[2].GetLogByName(log)
	#p3_up = readers[2].GetLogByName(log+'_up')
	#p3_down = readers[2].GetLogByName(log+'_down')
	#t3 = np.arange(1,len(p3)+1)
	
	plt.fill_between(t1, p1_down, p1_up,facecolor='blue', linewidth=0.0, alpha=0.5)
	plt1, = plt.plot(t1,p1,'b')

	#plt.fill_between(t2, p2_down, p2_up,facecolor='red', linewidth=0.0, alpha=0.5)
	#plt2, = plt.plot(t2,p2,'red')

	#plt.fill_between(t3, p3_down, p3_up,facecolor='green', linewidth=0.0, alpha=0.5)
	#plt3, = plt.plot(t3,p3,'green')

	#if log != 'estimated_value':
	#	plt.ylim((-200,370))
	#else:
	#	plt.ylim((-10,85))
	plt.xlabel('episode')
	plt.ylabel('reward')
	plt.title(lognames[log_index])
	#plt.legend([plt1, plt2, plt3], names, loc = 1)

	#if log == 'estimated_value':
	#	plt.show()
	fig.savefig(BASE+logs[log_index]+'.png')


