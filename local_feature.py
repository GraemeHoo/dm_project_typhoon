import numpy as np
import math
import os


def files_abspath():
	return ['all_data\\' + path for path in os.listdir('all_data\\')]


def format_data(file_path):
	data = [line.replace('\n', '').split() for line in open(file_path) if not line.startswith('6666')]
	index = []
	i = 0
	for line in open(file_path):
		if line.startswith('6666'):
			index.append(i)
			continue
		i += 1
	source_data = [data[i] for i in range(len(data)) if i in index]
	new_source_data = np.array([[d[1], d[2], d[3], d[4], d[5]] for d in source_data], dtype=int)
	return new_source_data


def source_data():
	paths = files_abspath()
	data = []
	for p in paths:
		data.append(format_data(p))
	data_np = data[0]
	for i in range(1, len(data)):
		data_np = np.vstack((data_np, data[i]))
	xy_data = data_np[:, 1:3]
	return xy_data


def local_feature_calculate(dataset):
	'''
	return distance, theta, delta_distance, delta_theta
	len(distance) = len(theta) = len(dataset) - 1
	len(delta_distance) = len(delta_theta) = len(dataset) - 2
	'''

	distance = []
	theta = []
	for i in range(len(dataset)):
		if i == 0:
			continue
		else:
			delta_latitude = dataset[i][0] - dataset[i - 1][0]
			delta_longitude = dataset[i][1] - dataset[i - 1][1]
			value = np.sqrt(delta_latitude ** 2 + delta_longitude ** 2)
			distance.append(value)
			if delta_longitude != 0:
				arctan_theta = np.abs(delta_latitude) / np.abs(delta_longitude)
			else:
				arctan_theta = 0
				print("the same longitude between the point", i, 'and ', i + 1)
			theta.append(math.atan(arctan_theta)*180/np.pi)

	delta_distance = []
	delta_theta = []
	for j in range(len(distance)):
		if j == 0:
			continue
		else:
			delta1 = distance[j] - distance[j - 1]
			delta_distance.append(delta1)

			delta2 = theta[j] - theta[j - 1]
			delta_theta.append(delta2)
	return distance, theta, delta_distance, delta_theta


if __name__ == '__main__':
	print((local_feature_calculate(source_data())))
