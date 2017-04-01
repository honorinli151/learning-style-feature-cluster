#! /user/bin/env python
# coding=utf-8
# Reference: http://scikit-learn.org/stable/modules/clustering.html

from numpy import *
from sklearn import cluster, datasets
from src import visualization


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
    # cluster_centers = affPropCluster(data)
    # cluster_centers = meanShiftCluster(data)
    # cluster_centers = wardCluster(data)
    # cluster_centers = specClutCluster(data)
    # print("Cluster centers:")
    # print(cluster_centers)
    # print("Sum of distance")
    # print(cluster_inertia)
    visualization.drawGraph(data)


"""
K-Means
"""
def kmeansCluster(dataset):
    k_means = cluster.KMeans(n_clusters=3)
    k_means.fit(dataset)
    return k_means.cluster_centers_, k_means.inertia_


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


"""
Affinity Propagation
http://www.cnblogs.com/huadongw/p/4202492.html
"""
def affPropCluster(dataset):
    aff_prop = cluster.AffinityPropagation()
    aff_prop.fit(dataset)
    return aff_prop.cluster_centers_


"""
Mean Shift
http://www.cnblogs.com/liqizhou/archive/2012/05/12/2497220.html
Cluster = 1, not recommend
"""
def meanShiftCluster(dataset):
    bandwidth = cluster.estimate_bandwidth(dataset, quantile=0.2)
    mean_shift = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)
    mean_shift.fit(dataset)
    return mean_shift.cluster_centers_


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


"""
Ward???????????????????????????????????????????????????
Recursively merges the pair of clusters that minimally increases within-cluster variance.
http://scikit-learn.sourceforge.net/stable/modules/generated/sklearn.cluster.Ward.html
"""
def wardCluster(dataset):
    #children, n_components, n_leaves, parents, distances = cluster.ward_tree(dataset)
    #return children
    cluster.ward_tree(dataset)

