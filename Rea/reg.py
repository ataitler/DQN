import csv
import numpy as np

images = []
labels = []
i = 1
with open('data.csv', 'rb') as csvfile:
	reader = csv.reader(csvfile, delimiter=',', quotechar='|')
	for row in reader:
		l = len(row)
		x = row[:l-1]
		images = images + x
		y = [row[l-1]]
		labels = labels + y
		print i
		i=i+1
		if i > 30000:
			break

samples = len(labels)
images = np.array(images).reshape(samples,14)
images = images.astype(np.float)
labels = np.array(labels).reshape(samples, 1)
labels = labels.astype(np.float)
np.savez_compressed('npdata',X=images, Y=labels)




