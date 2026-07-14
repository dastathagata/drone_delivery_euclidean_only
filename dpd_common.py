import numpy as np
import pulp as lp
from sympy import *
import cvxpy as cp
import networkx as nx

from pulp import LpMinimize, LpMaximize, LpProblem, LpStatus, lpSum, LpVariable

dpd_debug_enable = 0

#returns the distance between two points in either euclidean or manhattan region
def distance(x1, x2, y1, y2):
    return np.sqrt(((x1 - x2) ** 2) + ((y1 - y2) ** 2))
    
"""
def eu_distance(u,v): #returns the square of euclidean distance
    d=u-v
    return (np.dot(d,np.transpose(d)))

def form_2(C,loc,xy,D,k,cap): #Simple Linear Program cap is allowed to exceed maximum individual demand
    pairs=[(i,j) for i in loc for j in range(k)]
    centre_d=np.zeros((len(loc),k))
    
    
    for (i,j) in pairs:
        #centre_d[i,j]=abs(xy[i][0]-C[j][0])+abs(xy[i][1]-C[j][1]) #for manahattan
        centre_d[i,j]=np.sqrt(eu_distance(xy[i],C[j])) #for euclidean
    prob = LpProblem(name="LSAP",sense=LpMinimize)
    
    x = LpVariable.dicts("lt",[(i,j) for (i,j) in pairs],lowBound=0.0)
       #x[i,j] is the amount of demand of location i met by jth tour
    prob += lpSum([centre_d[i,j]*(x[i,j]) for (i,j)  in pairs]),"obj"
"""

#Simple Linear Program cap is allowed to exceed maximum individual demand
def simple_lp(loc, target_loc_xy, target_demand, no_of_cluster, drone_cap, centroids): 
    pairs = [(i,j) for i in loc for j in range(no_of_cluster)]
    centre_d = np.zeros((len(loc), no_of_cluster))

    for (i,j) in pairs:
        centre_d[i,j] = distance(target_loc_xy[i][0], centroids[j][0], target_loc_xy[i][1], centroids[j][1])
    # LSAP: Linear Sum Assignment Problem
    lsap = lp.LpProblem(name="LSAP",sense=lp.LpMinimize)
    
    x = lp.LpVariable.dicts("lt", [(i,j) for (i,j) in pairs], lowBound=0.0)

#objective
    #x[i,j] is the amount of demand of location i met by jth tour
    lsap += lp.lpSum([centre_d[i,j]*(x[i,j]) for (i,j)  in pairs])

#constraints
    for j in range(no_of_cluster):
        lsap += lp.lpSum([x[i,j] for i in loc]) <= drone_cap
    for i in loc:
        lsap += lp.lpSum([x[i,j] for j in range(no_of_cluster)]) == target_demand[i]

    return(lsap, x)

#avoid duplicate entry of node in the cluster list
def is_node_already_included_in_cluster(node_included_in_cluster_array, node_number):
    for i in range(len(node_included_in_cluster_array)):
        if (node_included_in_cluster_array[i] == node_number):
            return False
        
    return True

def geometric_median(target_loc_xy):
    n = len(target_loc_xy)
    d = cp.Variable(n)
    c = cp.Variable(2)
    prob = cp.Problem(cp.Minimize(cp.sum(d)),
                 [d[i]>=cp.norm(target_loc_xy[i]-c,2) for i in range(n)])

    prob.solve(verbose=False)
    dp = (c[0].value, c[1].value)

    return dp

def dist_matrix(target_loc_xy): #DISTANCE MATRIX
    n = len(target_loc_xy)
    d = np.zeros((n, n))
    for i in range(len(target_loc_xy)):
        for j in range(len(target_loc_xy)):
            d[i,j] = distance(target_loc_xy[i][0], target_loc_xy[j][0], target_loc_xy[i][1], target_loc_xy[j][1])
    return d

def CreateGraph(X, D):
    edges = []
    G = nx.Graph()
    n = len(X)
    for i in range(n):
        for j in range(i+1,n):
            e = (i,j,D[i,j])
            edges.append(e)
    G.add_weighted_edges_from(edges)

    return (G)