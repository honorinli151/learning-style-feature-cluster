#! /user/bin/env python
# coding=utf-8

"""
This file concludes 3 k-mean methods (k-means, bisecting k-means, k-means++)
"""

from numpy import *
import math

"""Load raw data set"""
class LoadData:

    @staticmethod
    def loadDataSet():  # general function to parse tab -delimited floats
        dataMat = []  # assume last column is target value
        fr = open('data/data.csv')  # 'data/data.csv'
        fr.readline()
        for line in fr.readlines():
            curLine = line.strip().split(',')
            fltLine = map(float, curLine)  # map all elements to float()
            dataMat.append(fltLine)
        return array(dataMat)

"""Calculate the distance between two points"""
class DistMeas:

    @staticmethod
    def distEclud(vecA, vecB):
        return sqrt(sum(power(vecA - vecB, 2)))  # la.norm(vecA-vecB)

"""Generate centroids"""
class Centroids:

    """Generate Random centroids"""
    @staticmethod
    def randCent(dataSet, k):
        n = shape(dataSet)[1]
        centroids = mat(zeros((k, n)))  # create centroid mat
        for j in range(n):  # create random cluster centers, within bounds of each dimension
            minJ = min(dataSet[:, j])
            rangeJ = float(max(dataSet[:, j]) - minJ)
            centroids[:, j] = mat(minJ + rangeJ * random.rand(k, 1))
        return centroids

    """Generate K-means++ centroids"""
    @staticmethod
    def kppCent(dataSet, k):
        """Picks a random point to serve as the first centroid"""
        n = shape(dataSet)[1]
        centroid_list = mat(zeros((k, n)))
        centroid_index = []
        centroid_count = 0
        index = random.randint(0, shape(dataSet)[0] - 1)
        centroid_list[centroid_count] = dataSet[index]
        centroid_count = 1
        centroid_index.append(index)  # Removes point associated with given index
        """Finds the other k-1 centroids from the remaining lists of points"""
        while centroid_count < k:
            distance_list = []
            for data in dataSet:
                min_distance = inf
                for centroid in centroid_list:
                    distance = DistMeas.distEclud(centroid, data)
                    if distance < min_distance:
                        min_distance = distance
                distance_list.append(min_distance)
            """Calculate the weighted probability"""
            distance_list = [x ** 2 for x in distance_list]
            dist_sum = sum(distance_list)
            weighted_list = [x / dist_sum for x in distance_list]
            indices = [i for i in range(len(distance_list))]
            chosen_index = random.choice(indices, p=weighted_list)
            for index in centroid_index:
                if chosen_index == index: continue
            centroid_list[centroid_count] = dataSet[index]
            centroid_index.append(chosen_index)
            centroid_count += 1
        return array(centroid_list)

class KMeans:

    def __init__(self, dataSet, k):
        self.dataSet = dataSet
        self.k = k

    def cluster(self, distMeas=DistMeas.distEclud, createCent=Centroids.randCent):
        m = shape(self.dataSet)[0]
        clusterAssment = mat(zeros((m, 2)))  # create mat to assign data points
        # to a centroid, also holds SE of each point
        centroids = createCent(self.dataSet, self.k)
        clusterChanged = True
        while clusterChanged:
            clusterChanged = False
            for i in range(m):  # for each data point assign it to the closest centroid
                minDist = inf
                minIndex = -1
                for j in range(self.k):
                    distJI = distMeas(centroids[j, :], self.dataSet[i, :])
                    if distJI < minDist:
                        minDist = distJI
                        minIndex = j
                if clusterAssment[i, 0] != minIndex: clusterChanged = True
                clusterAssment[i, :] = minIndex, minDist ** 2
            print centroids
            for cent in range(self.k):  # recalculate centroids
                ptsInClust = self.dataSet[nonzero(clusterAssment[:, 0].A == cent)[0]]  # get all the point in this cluster
                centroids[cent, :] = mean(ptsInClust, axis=0)  # assign centroid to mean
        return centroids, clusterAssment


class BiKMeans:

    def __init__(self, dataSet, k):
        self.dataSet = dataSet
        self.k = k

    def cluster(self, distMeas=DistMeas.distEclud):
        m = shape(self.dataSet)[0]
        clusterAssment = mat(zeros((m, 2)))
        centroid0 = mean(self.dataSet, axis=0).tolist()[0]
        centList = [centroid0]  # create a list with one centroid
        for j in range(m):  # calc initial Error
            clusterAssment[j, 1] = distMeas(mat(centroid0), self.dataSet[j, :]) ** 2
        while (len(centList) < self.k):
            lowestSSE = inf
            for i in range(len(centList)):
                ptsInCurrCluster = self.dataSet[nonzero(clusterAssment[:, 0].A == i)[0], :]  # get the data points currently in cluster i
                k = KMeans(ptsInCurrCluster, 2)
                centroidMat, splitClustAss = k.cluster(distMeas)
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
            clusterAssment[nonzero(clusterAssment[:, 0].A == bestCentToSplit)[0], :] = bestClustAss  # reassign new clusters, and SSE
        return mat(centList), clusterAssment

class KMeansPlusPlus:

    """Input is a 2D list of n-dimensional points"""
    def __init__(self, dataSet, k):
        self.dataSet = dataSet
        self.k = k

    def cluster(self):
        kpp = KMeans(self.dataSet, self.k)
        kpp.cluster(DistMeas.distEclud, Centroids.kppCent)

