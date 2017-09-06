import numpy as np
import matplotlib.pyplot as plt
from Logger import Logger

class Visualizer():
	def __init__(self, file_name):
		self.file_name = file_name
		self.Log = Logger()
		self.isLoaded = False
		#is_empty = self.Log.Load(self.file_name)
		#if is_empty:
		#	print "Log file is empty"
		#else:
		#	print "Log file loaded"

	def Load(self,file_name=None):
		if file_name is not None:
			self.file_name = file_name
		is_empty = self.Log.Load(self.file_name)
		if is_empty:	
			print "Log file is empty"
		else:
			print "Log file loaded"
			self.isLoaded = True

	def List(self):
		if not self.isLoaded:
			self.Load()
		logs = self.Log.ListLogs()
		print logs

	def ShowLog(self, *log_names):
		self.Load()
		plt.figure()
		logs = list(log_names)
		name = ""
		#if type(log_names) is list:
		#	name = ""
		for log in logs:
			name = name + log + " & "
			l = self.Log.GetLogByName(log)
			x_axis = np.arange(l.size)
			plt.plot(x_axis, l)
		plt.title(name[:-2])
		plt.xlabel('episodes')
		plt.show()

	def GetLog(self, log_name):
		self.Load()
		log = self.Log.GetLogByName(log_name)
		t = np.arange(log.size)
		return t, log

	def ShowSlicedLog(self, log_name, max_index):
		self.Load()
		max_index = max_index - 1
		log = self.Log.GetLogByName(log_name)
		t = np.arange(log.size)
		if max_index > log.size-1:
			max_index = log.size-1

		plt.plot(t[0:max_index],log[0:max_index])
		plt.title(log_name)
		plt.show()

	def ShowMulLog(self, *log_names):
		self.Load()
		plt.figure()
		logs = list(log_names)
		num_logs = len(logs)
		if num_logs < 3:
			cols = num_logs
			rows = 1
		else:
			cols = 3
			if num_logs % 3 == 0:
				rows = num_logs / 3
			else:
				rows = num_logs / 3 + 1
		c=1
		for i in xrange(1, rows+1):
			for j in xrange(1, cols+1):
				if c > num_logs:
					break
				plot_num = str(rows)+str(cols) + str(c)
				plt.subplot(int(plot_num))
				l = self.Log.GetLogByName(logs[c-1])
				x_axis = np.arange(l.size)
				plt.plot(x_axis,l)
				plt.title(logs[c-1])
				plt.xlabel('episodes')
				c = c+1
		plt.show()


	def ShowMulLogSliced(self, max_i, *log_names):
		self.Load()
		plt.figure()
		logs = list(log_names)
		num_logs = len(logs)
		if num_logs < 3:
			cols = num_logs
			rows = 1
		else:
			cols = 3
			if num_logs % 3 == 0:
				rows = num_logs / 3
			else:
				rows = num_logs / 3 + 1
		c=1
		for i in xrange(1, rows+1):
			for j in xrange(1, cols+1):
				if c > num_logs:
					break
				plot_num = str(rows)+str(cols) + str(c)
				plt.subplot(int(plot_num))
				l = self.Log.GetLogByName(logs[c-1])
				end_i = min(l.size, max_i)
				x_axis = np.arange(end_i)
				plt.plot(x_axis,l[0:end_i])
				plt.title(logs[c-1])
				plt.xlabel('episodes')
				c = c+1
		plt.show()




