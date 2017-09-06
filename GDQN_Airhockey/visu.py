from utils.Visualizer import Visualizer as V
import numpy as np
import matplotlib.pyplot as plt

v1 = V('hockey_multinet1_decay_rate1.2_2_V2/logs')
v2 = V('hockey_multinet1_decay_rate1.2_2_V3/logs')
V = [v1, v2]

V_all = []
log = 'network_middle'
for i in xrange(len(V)):
	t = V[i].GetLog(log)[0]
	v_graph = V[i].GetLog(log)[1]
	v_graph = v_graph.reshape(1,v_graph.size)
	if len(V_all) == 0:
		V_all = v_graph
	else:
		V_all = np.concatenate((V_all,v_graph),0)
y = V_all.mean(0)
err = np.std(V_all,0)
plt.fill_between(t, y+err, y-err)
plt.plot(t,y,'b-')
plt.show()


#v1.ShowMulLogSliced(100000,'network_left','network_middle','network_right','error','estimated_value','network_random')
