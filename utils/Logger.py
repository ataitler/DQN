from collections import deque
import numpy as np
import os

class Logger():
	def __init__(self):
		self.logs = deque()
		self.names = deque()

	def AddNewLog(self, name):
		if self.names.count(name) == 0:
			newlog = deque()
			self.logs.append(newlog)
			self.names.append(name)
			return True
		else:
			return False
	
	def AddRecord(self, logname, value):
		i = self._index(logname)
		if i == -1:
			return False
		else:
			self.logs[i].append(value)
			return True

	def ListLogs(self):
		return np.array(self.names)

	def GetLogByName(self, logname):
		i = self._index(logname)
		if i == -1:
			return None
		else:
			return np.array(self.logs[i])
	
	def RemoveLogByName(self, logname):
		i = self._index(logname)
		if i == -1:
			return False
		else:
			self.names.rotate(-i)
			self.names.popleft()
			self.logs.rotate(-i)
			self.logs.popleft()
			return True

	def Save(self, file_name):
		to_save = deque()
		for log in self.logs:
			to_save.append(np.array(log))
		np.array(to_save)
		np.savez_compressed(file_name, logs = np.array(to_save), names=np.array(self.names))
	
	def Load(self, file_name):
		if not os.path.exists(file_name+'.npz'):
			return False
		temp = np.load(file_name+".npz")
		logs_array = temp['logs']
		names_array = temp['names']
		self.logs = deque()
		for log in logs_array:
			self.logs.append(deque(log))
		self.names = deque(names_array)
	
	def _index(self, logname):
		for i, ele in enumerate(self.names):
			if ele == logname:
				return i
		else:
			return -1


