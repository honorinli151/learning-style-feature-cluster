#! /user/bin/env python
# coding=utf-8

from numpy import *
from sklearn import cluster, datasets, metrics

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

"""
Birch http://www.cnblogs.com/pinard/p/6200579.html
"""
def BirchCluster(dataset):
    birch = cluster.Birch(n_clusters=3)
    birch.fit(data)
    return birch.subclusters_centers_, birch.labels

def main():
    data = loadDataSet('data/data.cv')
    cluster_centers, cluster_labels = birch(data)
    Calinski_Harabasz = metrics.calinski_harabaz_score(data, birch.fit(data))
    print('Cluster_centers:')
    print(cluster_centers)
    print('Cluster_labels')
    print(cluster_labels)
    print('Calinski_Harabasz Score')
    print(Calinski_Harabasz)

"""
DBSCAN http://www.cnblogs.com/pinard/p/6208966.html
"""
def DBSCANCluster(dataset):
    dbscan = cluster.dbscan()
    dbscan.fit(data)
    return dbscan.labels_

def main():
    data = loadDataSet('data/data.csv')
    cluster_labels = DBSCANCluster(data)
    n_clusters = len(set(cluster_labels)) - (1 if -1 in labels else 0)
    completeness = metrics.completeness_score(cluster_labels_true, cluster_labels) #The percentage of data rest
    Calinski_Harabasz = metrics.calinski_harabaz_score(data, dbscan.fit(data))
    print('n_clusters;')
    print(n_clusters)
    print('Completeness')
    print(completeness)
    print('Calinski_Harabasz')
    print(Calinski_Harabasz)

"""
Hierarchical clustering
"""
def HClustering(dataset):
    hclustering = cluster.AgglomerativeClustering(n_clusters=3)
    hclustering.fit()
    return hclustering.labels_

def main():
    data = loadDataSet('data/data.csv')
    cluster_labels = HClustering(data)
    n_clusters = len(set(cluster_labels)) - (1 if -1 in labels else 0)
    Calinski_Harabasz = metrics.calinski_harabaz_score(data, dbscan.fit(data))
    print('n_clusters;')
    print(n_clusters)
    print('Calinski_Harabasz')
    print(Calinski_Harabasz)


