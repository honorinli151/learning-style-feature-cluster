#! /user/bin/env python
# coding=utf-8

"""
This file concludes 3 k-mean methods (k-means, bisecting k-means, k-means++)
"""

from numpy import *
import random
import math

class LoadData:

    @staticmethod
    def loadDataSet(filepath):  # general function to parse tab -delimited floats
        dataMat = []  # assume last column is target value
        fr = open(filepath)  # 'data/data.csv'
        fr.readline()
        for line in fr.readlines():
            curLine = line.strip().split(',')
            fltLine = map(float, curLine)  # map all elements to float()
            dataMat.append(fltLine)
        return array(dataMat)


def distEclud(vecA, vecB):
    return sqrt(sum(power(vecA - vecB, 2)))  # la.norm(vecA-vecB)


def randCent(dataSet, k):
    n = shape(dataSet)[1]
    centroids = mat(zeros((k, n)))  # create centroid mat
    for j in range(n):  # create random cluster centers, within bounds of each dimension
        minJ = min(dataSet[:, j])
        rangeJ = float(max(dataSet[:, j]) - minJ)
        centroids[:, j] = mat(minJ + rangeJ * random.rand(k, 1))
    return centroids


def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m, 2)))  # create mat to assign data points
    # to a centroid, also holds SE of each point
    centroids = createCent(dataSet, k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):  # for each data point assign it to the closest centroid
            minDist = inf;
            minIndex = -1
            for j in range(k):
                distJI = distMeas(centroids[j, :], dataSet[i, :])
                if distJI < minDist:
                    minDist = distJI;
                    minIndex = j
            if clusterAssment[i, 0] != minIndex: clusterChanged = True
            clusterAssment[i, :] = minIndex, minDist ** 2
        print centroids
        for cent in range(k):  # recalculate centroids
            ptsInClust = dataSet[nonzero(clusterAssment[:, 0].A == cent)[0]]  # get all the point in this cluster
            centroids[cent, :] = mean(ptsInClust, axis=0)  # assign centroid to mean
    return centroids, clusterAssment


def biKmeans(dataSet, k, distMeas=distEclud):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m, 2)))
    centroid0 = mean(dataSet, axis=0).tolist()[0]
    centList = [centroid0]  # create a list with one centroid
    for j in range(m):  # calc initial Error
        clusterAssment[j, 1] = distMeas(mat(centroid0), dataSet[j, :]) ** 2
    while (len(centList) < k):
        lowestSSE = inf
        for i in range(len(centList)):
            ptsInCurrCluster = dataSet[nonzero(clusterAssment[:, 0].A == i)[0],
                               :]  # get the data points currently in cluster i
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas)
            sseSplit = sum(splitClustAss[:, 1])  # compare the SSE to the currrent minimum
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:, 0].A != i)[0], 1])
            print "sseSplit, and notSplit: ", sseSplit, sseNotSplit
            if (sseSplit + sseNotSplit) < lowestSSE:
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit
        bestClustAss[nonzero(bestClustAss[:, 0].A == 1)[0], 0] = len(centList)  # change 1 to 3,4, or whatever
        bestClustAss[nonzero(bestClustAss[:, 0].A == 0)[0], 0] = bestCentToSplit
        print 'the bestCentToSplit is: ', bestCentToSplit
        print 'the len of bestClustAss is: ', len(bestClustAss)
        centList[bestCentToSplit] = bestNewCents[0, :].tolist()[0]  # replace a centroid with two best centroids
        centList.append(bestNewCents[1, :].tolist()[0])
        clusterAssment[nonzero(clusterAssment[:, 0].A == bestCentToSplit)[0],
        :] = bestClustAss  # reassign new clusters, and SSE
    return mat(centList), clusterAssment

class K_Means_Plus_Plus:

    """Input is a 2D list of n-dimensional points"""
    def __init__(self, points_list, k):
        self.centroid_count = 0
        self.point_count = len(points_list)
        self.cluster_count = k
        self.points_list = list(points_list)
        self.initialize_random_centroid()
        self.initialize_other_centroids()

    """Picks a random point to serve as the first centroid"""
    def initialize_random_centroid(self):
        self.centroid_list = []
        index = random.randint(0, len(self.points_list)-1)

        self.centroid_list.append(self.remove_point(index))
        self.centroid_count = 1

    """Removes point associated with given index so it cannot be picked as a future centroid.
    Returns list containing coordinates of newly removed centroid"""
    def remove_point(self, index):
        new_centroid = self.points_list[index]
        del self.points_list[index]

        return new_centroid

    """Finds the other k-1 centroids from the remaining lists of points"""
    def initialize_other_centroids(self):
        while not self.is_finished():
            distances = self.find_smallest_distances()
            chosen_index = self.choose_weighted(distances)
            self.centroid_list.append(self.remove_point(chosen_index))
            self.centroid_count += 1


    """Calculates distance from each point to its nearest cluster center. Then chooses new
    center based on the weighted probability of these distances"""
    def find_smallest_distances(self):
        distance_list = []

        for point in self.points_list:
            distance_list.append(self.find_nearest_centroid(point))

        return distance_list

    """Finds centroid nearest to the given point, and returns its distance"""
    def find_nearest_centroid(self, point):
        min_distance = math.inf

        for values in self.centroid_list:
            distance = self.euclidean_distance(values, point)
            if distance < min_distance:
                min_distance = distance

        return min_distance

    """Chooses an index based on weighted probability"""
    def choose_weighted(self, distance_list):
        distance_list = [x**2 for x in distance_list]
        weighted_list = self.weight_values(distance_list)
        indices = [i for i in range(len(distance_list))]
        return np.random.choice(indices, p = weighted_list)

    """Weights values from [0,1]"""
    def weight_values(self, list):
        sum = np.sum(list)
        return [x/sum for x in list]

    """computes N-d euclidean distance between two points represented as lists:
     (x1, x2, ..., xn) and (y1, y2, ..., yn)"""
    def euclidean_distance(self, point1, point2):
        point1 = np.asarray(point1)
        point2 = np.asarray(point2)

        return np.linalg.norm(point2-point1)

    """Checks to see if final condition has been satisfied (when K centroids have been created)"""
    def is_finished(self):
        outcome = False
        if self.centroid_count == self.cluster_count:
            outcome = True

        return outcome

    """Returns final centroid values"""
    def final_centroids(self):
        return self.centroid_list