#! /user/bin/env python
# coding=utf-8

# List of imports.
import numpy as np
from numpy import linalg
from numpy.linalg import norm
from scipy.spatial.distance import squareform, pdist

# We import sklearn.
import sklearn
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits
from sklearn.preprocessing import scale

# We'll hack a bit with the t-SNE code in sklearn 0.15.2.
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.manifold.t_sne import (_joint_probabilities, _kl_divergence)
from sklearn.utils.extmath import _ravel

# Random state.
RS = 20150101

# We'll use matplotlib for graphics.
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects

# We import seaborn to make nice plots.
import seaborn as sns
sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})


"""
Draw Graph
"""
def drawGraph(data):
    data_proj = tSNEEmbedding(data)
    plt.scatter(data_proj[:, 0],data_proj[:, 1])
    plt.savefig('data/OriginalData.png')


"""
t-SNE
Reference: http://mtpgm.com/2015/08/17/t-sne/
"""
def tSNEEmbedding(dataset):
    tsne = TSNE()
    return tsne.fit_transform(dataset)


