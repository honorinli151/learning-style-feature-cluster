#! /user/bin/env python
# coding=utf-8
# Reference: http://scikit-learn.org/stable/modules/clustering.html

from numpy import *
from sklearn import cluster
from src import visualization
from src import cluster_evaluation

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
MAIN
"""
def main():
    data = loadDataSet('data/data.csv')
    # cluster_centers, cluster_inertia = kmeansCluster(data)
    # cluster_inertia = kmeansCluster(data)
    # labels = affPropCluster(data)
    # labels = meanShiftCluster(data)
    # cluster_centers = wardCluster(data)
    # labels = specClutCluster(data)
    # labels = DBSCANCluster(data)
    # labels = BirchCluster(data)
    # labels = AClustering(data)
    # print("Cluster centers:")
    # print(cluster_centers)
    # print("Sum of distance")
    # print(cluster_inertia)
    # print "Cluster number: ", unique(labels).shape[0]
    # cluster_evaluation.evaluation(data, labels)
    # visualization.drawGraph(data, labels, algo="AC", decompmethod="PCA")


"""
K-Means
"""
def kmeansCluster(dataset):
    inertias = zeros(20)
    for i in range(2, 22):
        k_means = cluster.KMeans(n_clusters=i)
        k_means.fit(dataset)
        inertias[i-2] = k_means.inertia_
    # return k_means.cluster_centers_, k_means.inertia_
    # return k_means.labels_
    return inertias


"""
Birch
http://www.cnblogs.com/pinard/p/6200579.html
"""
def BirchCluster(dataset):
    birch = cluster.Birch(n_clusters=8)
    birch.fit(dataset)
    return birch.labels_


"""
AgglomerativeClustering
"""
def AClustering(dataset):
    aclustering = cluster.AgglomerativeClustering(n_clusters=8)
    aclustering.fit(dataset)
    return aclustering.labels_


"""
Ward???????????????????????????????????????????????????
Recursively merges the pair of clusters that minimally increases within-cluster variance.
http://scikit-learn.sourceforge.net/stable/modules/generated/sklearn.cluster.Ward.html
"""
def wardCluster(dataset):
    #children, n_components, n_leaves, parents, distances = cluster.ward_tree(dataset)
    #return children
    cluster.ward_tree(dataset)


"""
DBSCAN 
http://www.cnblogs.com/pinard/p/6208966.html
"""
def DBSCANCluster(dataset):
    dbscan = cluster.DBSCAN()
    dbscan.fit(dataset)
    return dbscan.labels_


"""
Affinity Propagation
http://www.cnblogs.com/huadongw/p/4202492.html
"""
def affPropCluster(dataset):
    aff_prop = cluster.AffinityPropagation()
    aff_prop.fit(dataset)
    return aff_prop.labels_


"""
Mean Shift
http://www.cnblogs.com/liqizhou/archive/2012/05/12/2497220.html
Cluster = 1, not recommend
"""
def meanShiftCluster(dataset):
    # bandwidth = cluster.estimate_bandwidth(dataset, quantile=0.2)
    mean_shift = cluster.MeanShift()
    mean_shift.fit(dataset)
    return mean_shift.labels_


"""
Spectral Clustering
(K-Means + PCA)
http://www.cnblogs.com/sparkwen/p/3155850.html
https://www.52ml.net/12180.html
"""
def specClutCluster(dataset):
    spec_clut = cluster.SpectralClustering()
    spec_clut.fit(dataset)
    return spec_clut.labels_

