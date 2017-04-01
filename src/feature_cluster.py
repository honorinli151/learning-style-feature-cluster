#! /user/bin/env python
# coding=utf-8

from numpy import *
from sklearn import cluster, datasets

"""Load raw data set"""
def loadDataSet(filename):
    dataMat = []
    fr = open(filename)
    fr.readline()
    for line in fr.readlines():
        curLine = line.strip().split(',')
        fltLine = map(float, curLine)  # map all elements to float()
        dataMat.append(fltLine)
    return array(dataMat)

"""
K-Means
"""
def kmeansCluster(dataset):
    k_means = cluster.KMeans(n_clusters=3)
    k_means.fit(dataset)
    return k_means.cluster_centers_, k_means.inertia_

def main():
    data = loadDataSet('data/data.csv')
    cluster_centers, cluster_inertia = kmeansCluster(data)
    print("Cluster centers:")
    print(cluster_centers)
    print("Sum of distance")
    print(cluster_inertia)
