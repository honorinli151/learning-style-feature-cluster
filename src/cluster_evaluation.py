#! /user/bin/env python
# coding=utf-8

from sklearn import metrics

def evaluation(dataset, labels):
    sc_score = scEvaluation(dataset, labels)
    ch_score = chEvaluation(dataset, labels)
    print "Silhouette Coefficient: ", sc_score
    print "Calinski-Harabaz Index: ", ch_score

"""
Silhouette Coefficient
http://scikit-learn.org/stable/modules/clustering.html#silhouette-coefficient
"""
def scEvaluation(dataset, labels):
    return metrics.silhouette_score(dataset, labels, metric='euclidean')

"""
Calinski-Harabaz Index
http://scikit-learn.org/stable/modules/clustering.html#calinski-harabaz-index
"""
def chEvaluation(dataset, labels):
    return metrics.calinski_harabaz_score(dataset, labels)