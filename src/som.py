

"""
This file concludes SOM methods
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

"""
One cluster methods: SOM
"""
class Nodes

    """generate random nodes for algorithm SOM"""
    @staticmethod
    def ranNodes(dataSet,k):  # caution!!!! here k must be a number extreme small in front of n
        n = shape(dataSet)[1]
        NodesList = random.random(k,n)
    return NodesList

class SOM

    """Input is a 2D list of n-dimensional points"""
    def __init__(self):
        self.dataSet = dataSet
        self.k = k

    def cluster(self,dataSet,k):

        """Initializer the matrix of dataSet by norming"""
        inidataSet = (1/40) * dataSet

        """Associate data points with nodes and Calculate for every nodes the of Dist"""
        m = shape(dataSet)[1]
        clusterAssment = mat(zeros((m, 2)))
        distsumnodes = mat(zeros((k,1)))
            for i in range(m):  # for each data point assign it to the closest nodes
                minDist = inf
                minIndex = -1
                for j in range(self.k):
                    distJI = distMeas(NodesList[j, :], inidataSet[i, :])
                    if distJI < minDist:
                        minDist = distJI
                        minIndex = j
                clusterAssment[i, :] = minIndex, minDist ** 2
                BMU = argmax(distsumnodes)
                for j in range(self.k):
                    NodesList[j,;] += 0.6 * pow(math.e,(-(distMeas(NodesList[j,:],NodesList[minIndex,:]) ** 2)/8))


    


         





