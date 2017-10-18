from numpy import *
import math
import random
import os
import matplotlib.pyplot as plt

iterate = 5000

def files_abspath():
	return ['alldata\\' + path for path in os.listdir('alldata')]


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
	new_source_data = array([[d[1], d[2], d[3], d[4], d[5]] for d in source_data], dtype=float)
	return new_source_data


# 计算两个样本距离_欧几里得距离
def euclindeanDistance(instance1, instance2, col):
	distance = 0
	for x in range(col):
		distance += pow(instance1[:, x] - instance2[:, x], 2)
	return math.sqrt(distance)

# 随机生成初始的质心
def initCentroids(dataSet, k):
	numSamples, dim = dataSet.shape
	centroids = mat(zeros((k, dim)))
	index = random.sample(range(numSamples), k)
	for i in range(k):
		centroids[i, :] = dataSet[index[i], :]
	return centroids


def kMeans(dataSet, k, distMeas= euclindeanDistance, createCent=initCentroids):
	row, col = dataSet.shape

	clusterAssment = mat(zeros((row, col)))  # create mat to assign data points
	# to a centroid, also holds SE of each point
	centroids = createCent(dataSet, k)
	clusterChanged = True
	iter = 0
	while clusterChanged & (iter < iterate):
		clusterChanged = False
		iter = iter + 1
		for i in range(row):  # for each data point assign it to the closest centroid
			minDist = inf
			minIndex = -1
			tempDist = list(zeros(k))
			for j in range(k):
				tempDist[j] = euclindeanDistance(centroids[j, :], dataSet[i, :], col)
			minDist = min(tempDist)
			minIndex = tempDist.index(minDist)

			if clusterAssment[i, 0] != minIndex:
				clusterChanged = True
			clusterAssment[i, :] = minIndex, minDist ** 2
		print(centroids)

		for cent in range(k):  # recalculate centroids
			ptsInClust = dataSet[nonzero(clusterAssment[:, 0].A == cent)[0]]  # get all the point in this cluster
			centroids[cent, :] = mean(ptsInClust, axis=0)  # assign centroid to mean
	return centroids, clusterAssment


def show(dataSet, k, centroids, clusterAssment):
	numSamples, dim = dataSet.shape
	mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
	for i in range(numSamples):
		markIndex = int(clusterAssment[i, 0])
		plt.plot(dataSet[i, 0], dataSet[i, 1], mark[markIndex])
	mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']
	for i in range(k):
		plt.plot(centroids[i, 0], centroids[i, 1], mark[i], markersize=12)
	plt.show()

def source_data():
	paths = files_abspath()
	data = []
	for p in paths:
		data.append(format_data(p))
	data_np = data[0]
	for i in range(1, len(data)):
		data_np = vstack((data_np, data[i]))
	xy_data = data_np[:, 1:3]
	return xy_data

def main():
	dataset = source_data()
	dataset[:, 1] = dataset[:, 1] * 0.1
	dataSet = mat(dataset)
	k = 10
	centroids, clusterAssment = kMeans(dataSet, k)
	show(dataSet, k, centroids, clusterAssment)

if __name__ == '__main__':
	main()
