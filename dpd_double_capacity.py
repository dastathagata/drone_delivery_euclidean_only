# importing the module
import numpy as np
import dpd_common as dpd_comm
import dpd_single_capacity as dpd_sc
import cvxpy as cp
from sympy import *
import networkx as nx

def Match(G,n,Y,D,X): #minimum weight match
    edges=[]
    matchedtoY=-1; #The point to which Y is mapped
    if (n % 2 != 0): #if n is odd
        for i in range(n):
            e = (i, n, dpd_comm.distance(X[i][0],Y[0], X[i][1], Y[1])) #add weight of edges from Y to others
            edges.append(e)
        G.add_weighted_edges_from(edges)
    M = nx.min_weight_matching(G)
    cost = 0
    #print (M)
    for m in M:
        if (m[0]==n):
            matchedtoY = X[m[1]]
        elif (m[1]==n):
            matchedtoY = X[m[0]]
        else:
            cost=cost + D[m[0],m[1]]

    return (cost, matchedtoY)

def double_cap_drone_total_traversal(target_loc_xy):
    dm = dpd_comm.dist_matrix(target_loc_xy)
    G = dpd_comm.CreateGraph(target_loc_xy, dm)
    n = len(target_loc_xy)
    Y = (0,0)

    dp = dpd_comm.geometric_median(target_loc_xy)
    d_e = dpd_sc.drone_totalpath_traversal_single_capacity(target_loc_xy, dp)
    #print("d_e: %d" % (d_e))

    if (n % 2 == 0):
        (d_M, anything) = Match(G,n,Y,dm,target_loc_xy)
    else:
        (d_M, target_loc_xy1) = Match(G,n,dp,dm,target_loc_xy) #X1 matched to dp if n is odd
        #print("Old", target_loc_xy)
        target_loc_xy = target_loc_xy.tolist()  # Convert to list
        target_loc_xy.append(target_loc_xy1)    # Now append works
        target_loc_xy = np.array(target_loc_xy)
        #print("New", target_loc_xy)
        #target_loc_xy.append(target_loc_xy1)
        dp = dpd_comm.geometric_median(target_loc_xy)
        #print ("updated GM", dp)
        d_e = dpd_sc.drone_totalpath_traversal_single_capacity(target_loc_xy, dp)
    
    total = d_e + d_M
    #print("total distance %d" % (total))
    
    return total