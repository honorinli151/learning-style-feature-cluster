#! /user/bin/env python
# coding=utf-8
# Reference: http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.scatter

# List of imports.
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


"""
Draw Graph
"""
def drawGraph(data, labels, filename):
    # data_proj = tSNEEmbedding(data)
    data_proj = pcaDecomposition(data)
    plt.scatter(data_proj[:, 0], data_proj[:, 1], s=30, c=1.0*labels)
    plt.savefig(filename)


"""
t-SNE
Reference: http://mtpgm.com/2015/08/17/t-sne/
"""
def tSNEEmbedding(dataset):
    tsne = TSNE()
    return tsne.fit_transform(dataset)


"""
PCA
"""
def pcaDecomposition(dataset):
    pca = PCA(n_components=2)
    return pca.fit_transform(dataset)