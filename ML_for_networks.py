#!/usr/bin/env python
# coding: utf-8
#Group: Satwant kaur, Camilo Hernandez-Toro, Pietro Gerletti

'''
Implement power iteration clustering (PIC) and apply the PIC algorithm to the 
Zachary’s Karate Club network
'''


# ### Task 1: Implement the Power Iteration Clustering algorithm

#importing libraries
import numpy as np
import networkx as nx
from pylab import *
import csv
import igraph as ig
from sklearn.metrics.cluster import adjusted_rand_score

# Import the dataset from networkx
G = nx.karate_club_graph()
# Original clustering obtained from https://www.learndatasci.com/tutorials k-means-clustering-algorithms-python-intro/
y_true = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

# define affinity matrix
A_mat = np.zeros(shape=(len(G),len(G)))
for node in G:
    for neighbor in G.neighbors(node):
        A_mat[node][neighbor] = 1
    A_mat[node][node] = 1

# Power Iteration clustering
def pic(a,maxiter):
    dia=np.matrix(np.diag(a.sum(0))).I # getting the inverse of diagonal from affinity matrix
    w=dia*a # getting the normalised affinity matrix
    n=w.shape[0]
    #v = v0 #initial vector
    v = np.matrix(np.random.rand(a.shape[0])).T #initial vector
    for i in range(maxiter):
        v2=w*v
        c=1.0/np.linalg.norm(v2)
        v2*=c
        delta=np.linalg.norm(v2-v)
        if (delta*n)<1.0e-5:
            print(i)
            break
        v=v2
    return v

# KMeans clustering on the vector obtained from PIC algorithm 
def km_cluster(v):
    from sklearn.cluster import KMeans
    from sklearn.metrics.cluster import adjusted_rand_score
    kmeans = KMeans(n_clusters=2, random_state=0).fit(v)
    y_pred = list(kmeans.labels_)
    return(y_pred,adjusted_rand_score(y_true, y_pred))
    print("labels: ",y_pred)
    print(adjusted_rand_score(y_true, y_pred))

pos = nx.spring_layout(G)
# draw the graph copmparing the clustering labels with the originals
def draw_true_vs_pred(G, y_true, y_pred, pos):
    color_map =[]
    for i,j in enumerate(y_true):
        if j == y_pred[i]:
            color_map.append('blue')
            #node_shape = 'o'
        else:
            color_map.append('red')
            #node_shape = 'X'       
    # Draw edges
    nx.draw(G, pos,node_color = color_map, with_labels = True)
# draw the graph with some labels    
def draw_labels(G, y, pos):
    color_map =[]
    for l in y:
        if l:
            color_map.append('blue')
            #node_shape = 'o'
        else:
            color_map.append('red')
            #node_shape = 'X'
    # Draw edges
    nx.draw(G, pos,node_color = color_map, with_labels = True)
# draw all graphs    
def draw(y,score):
    y_pred =y
    subplot(3,1,1)
    title('PIC vs original - Adjusted Rand Score: %.6s' % score)
    draw_true_vs_pred(G,y_true, y_pred, pos)
    subplot(3,1,2)
    title('PIC clustering')
    draw_labels(G, y_pred, pos)
    subplot(3,1,3)
    title('Original clustering')
    draw_labels(G, y_true, pos)

def PIC(a,iterations,fig=1,verbose=False):
    import matplotlib.pyplot as plt
    v = pic(a,iterations)
    result = km_cluster(v)    
    y_pred = result[0]
    score = result[1]
    if verbose == True:
        fig = plt.figure(fig)
        draw(y_pred,score)
    return(score)

PIC(A_mat,100,fig=1,verbose=True)


# %% TASK 2

'''
Apply the PIC algorithm on two real biological networks (BioSnap) and compare the results
with Clauset-Newman-Moore greedy modularity maximization - another community detection algorithm 

'''

# Opening .csv file downloaded from BioSnap

network = []

file_name = "ChCh-Miner_durgbank-chem-chem.tsv"
csv.register_dialect('myDialect', delimiter = '\t')

with open(file_name, 'r') as csvfile:
    reader = csv.reader(csvfile, dialect='myDialect')
    for row in reader:
        network.append(row)
csvfile.close()

network = np.array(network)


## Creating Adjacency matrix and Graph
#
## Names of nodes in the network
#nodes = np.unique(network)
## Number of nodes
#n = len(nodes)
## Adjacency matrix
#A1 = np.zeros((n,n))
## Create graph
#gg1 = nx.Graph()
#
## Filling adjacency matrix with the interactions
#for inn in network:
#    a = inn[0]
#    b = inn[1]
#    ia = np.where(nodes == a)[0][0]
#    ib = np.where(nodes == b)[0][0]
#    A1[ia,ib] = 1
#    # Add interactions as edges in the graph
#    gg1.add_edge(a,b)
## Include interactions with self in Adjacency matrix
#A1+= np.identity(n)
#

## Identifying communities using Clauset-Newman-Moore greedy modularity maximization.
#
#from networkx.algorithms.community import greedy_modularity_communities
#c1 = list(greedy_modularity_communities(gg1))

# %% Creating Adjacency matrix

# Names of nodes in the network
nodes = np.unique(network)
# Number of nodes
n = len(nodes)
# Adjacency matrix
A1 = np.zeros((n,n))
# Create graph
#gg1 = nx.Graph()

gg1 = ig.Graph()
gg1.add_vertices(n)

# Fillinf adjacency matrix with the interactions
for inn in network:
    a = inn[0]
    b = inn[1]
    ia = np.where(nodes == a)[0][0]
    ib = np.where(nodes == b)[0][0]
    A1[ia,ib] = 1
    # Add interactions as edges in the graph
    gg1.add_edges([(ia,ib)])

A1+= np.identity(n)

# %%
dendrogram = gg1.community_edge_betweenness()
clust_f_g = dendrogram.as_clustering(n=3)
clustt = clust_f_g.membership
# In[]
# Opening .csv file downloaded from BioSnap

network = []

file_name = "ChG-Miner_miner-chem-gene.tsv"
csv.register_dialect('myDialect', delimiter = '\t')

with open(file_name, 'r') as csvfile:
    reader = csv.reader(csvfile, dialect='myDialect')
    for row in reader:
        network.append(row)
csvfile.close()

network = np.array(network)


# Creating Adjacency matrix and Graph

# Names of nodes in the network
nodes = np.unique(network)
# Number of nodes
n = len(nodes)
# Adjacency matrix
A2 = np.zeros((n,n))
# Create graph
gg2 = nx.Graph()

# Fillinf adjacency matrix with the interactions
for inn in network:
    a = inn[0]
    b = inn[1]
    ia = np.where(nodes == a)[0][0]
    ib = np.where(nodes == b)[0][0]
    A2[ia,ib] = 1
    # Add interactions as edges in the graph
    gg2.add_edge(a,b)
# Include interactions with self in Adjacency matrix
A2+= np.identity(n)


# Identifying communities using Clauset-Newman-Moore greedy modularity maximization.

from networkx.algorithms.community import greedy_modularity_communities
c2 = list(greedy_modularity_communities(gg2))


