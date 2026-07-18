# importing the module
import numpy as np
import dpd_common as dpd_comm
import dpd_single_capacity as dpd_sc
from sympy import *
import pulp as lp
import networkx as nx #needed for minimum weight matching

dpd_debug_enable = 0

def cost(xk,c):
    co=0;
    for i in range(len(xk)):
        d = dpd_comm.distance(xk[i][0], c[0], xk[i][1], c[1])
        co=co+d
    return (co)

# loc, target_loc_xy, x, no_of_cluster, target_demand
def clusterize(loc, xy, x, k, D):
    xc = []
    sol = np.zeros((len(xy),k))
    DC = np.zeros(k)
    DA = np.zeros(len(xy))
    for j in range(k):
        xk = []
        for i in loc:
            sol[i,j] = x[i,j].value()
            if (x[i,j].value() != 0):
                xk.append(xy[i])
                DC[j] = DC[j]+x[i,j].value()
                DA[i] = DA[i]+x[i,j].value()
            xc.append(xk)
    #print ("demand cluster-wize",DC) 
    #print ("demand satisfied",DA)
    ch = (D == DA)
    #print (ch)
    return(xc, sol)

def Centr(xc):
    C = []
    tc = 0
    tn = 0
    for i in range(len(xc)):
        xk = xc[i]
        #print (len(xk),i)
        Ci = np.sum(xk,axis=0)/len(xk)
        C.append(Ci)
        tc = tc + cost(xk,Ci)
        tn = tn + len(xk)
    #print ("total",tn)
    return(C, tc)

def furthest(x, c, k): #Find a new centroid furthest from the centroids c
    n=len(x)
    
    #print (n,k)
    d=np.zeros((n,k))
    dmin=np.zeros((n))
    for i in range(n):
        for j in range(k):
            d[i,j] = dpd_comm.distance(x[i][0], c[j][0], x[i][1], c[j][1])
        dmin[i]=min(d[i]) #minimum distance of x[i] from all in c
    l=np.argmax(dmin)
    #print ("next",l)
    return x[l]

def centroidi(x, k): #find a set of k centroids
    n=len(x)
    #print("len", n)
    d=x.shape[1]
    c=np.zeros((k,d))
    f=np.random.choice(n)
    c[0]=x[f]
    for i in range(k-1): #add k-1 new centroids
        nextc=furthest(x,c,i+1)
        c[i+1]=nextc
    return(c)

def Label(x, c): #find the nearest centroid c from each data and label them
    k=len(c)
    n=len(x)
    d=np.zeros((n,k))
    label=np.zeros((n))
    for i in range(n):
        for j in range(k):
            d[i,j] = dpd_comm.distance(x[i][0], c[j][0], x[i][1], c[j][1])
            
        label[i]=np.argmin(d[i])
    return label.astype(int)

def classify(x, demand, label): #determine clusters from labels
    xc=[]
    dc=[]
    for l in set(label):
        index=(label==l)
        xc.append(x[index])
        dc.append(demand[index])
        
    return (xc,dc) #xc is an array of size k, each having a cluster of points
    
def avg(x,d): #Find centroid of a cluster x 
    #Y=np.sum(x,axis=0)/len(x)
    Y=np.dot(d.T,x)/sum(d) #assuming multiplicity for demand>1
    return (Y)
         
#kmean clustering --> loc_xy: data point set, dem: demand set, k: Number of cluster    
def clustering(loc_xy, dem, k):
    C = centroidi(loc_xy, k)
    L1 = Label(loc_xy, C)
    for i in range(k):
        sse=0
        (XC, DC) = classify(loc_xy, dem, L1)
        for j in range(k):
            C[j] = avg(XC[j], DC[j])
        #print("iteration",i)     
        L2 = Label(loc_xy, C)
        change = (L2 != L1)
        if (sum(change) == 0):
            break
        L1 = L2
    return(C) 

def const_k_mean(loc, target_loc_xy, target_demand, no_of_cluster, drone_cap):
    centroids = clustering(target_loc_xy, target_demand, no_of_cluster)
    #print ("centroids", centroids)
    
    solver = lp.PULP_CBC_CMD(msg=False)

    n = len(target_loc_xy)
    sol = np.zeros((2, n, no_of_cluster))
    flag = True
    i = 1
    while flag:
        (lp_prob, x) = dpd_comm.simple_lp(loc, target_loc_xy, target_demand, no_of_cluster, drone_cap, centroids)
        status = lp_prob.solve(solver)
        if (status != 1):
            print("Linear programming is not solved. LP Status: ", lp.LpStatus[status])
            exit(1)
    
        (new_centroids, sol[i%2]) = clusterize(loc, target_loc_xy, x, no_of_cluster, target_demand)
        #print("new_centr", new_centroids)
        (C, tc) = Centr(new_centroids)
        change = (sol[i%2] != sol[(i-1)%2])
        i = i+1
        if (np.sum(change) == 0):
            flag=False

    centroids = np.array(centroids)
    return (centroids, x)

from scipy.spatial.distance import pdist, squareform

def chistofide_tsp(points):
    d_m = dpd_comm.dist_matrix(points)
    G = dpd_comm.CreateGraph(points, d_m)

    if len(G.nodes) == 0 or len(G.edges) == 0:
        print("Error: The graph is empty! Please check the input data.")
        exit()
    #else:
        #print("Graph is connected")
    
    tsp_tour = nx.approximation.traveling_salesman_problem(G, cycle=True)

    tour_distances = 0
    for i in range(len(tsp_tour) - 1):
        node1, node2 = tsp_tour[i], tsp_tour[i + 1]
        
        distance = dpd_comm.distance(points[node1][0], points[node2][0], points[node1][1], points[node2][1])
        
        tour_distances = tour_distances + distance

    return (tsp_tour, tour_distances)

def do_tsp(loc, no_of_cluster, target_loc_xy, target_demand, no_of_target, x):
    node_sequence_in_cluster_array=[]
    node_included_in_cluster_array=[]
    for j in range(no_of_cluster):
        node_sequence_single_cluster=[]
        for i in loc:
            if ((x[i,j].value()!=0) and (dpd_comm.is_node_already_included_in_cluster)(node_included_in_cluster_array, i)):
                node_sequence_single_cluster.append(i)
                node_included_in_cluster_array.append(i)
        xnj = target_loc_xy[node_sequence_single_cluster] #node sequence for jth tour
        node_sequence_in_cluster_array.append(xnj)
    #print("node_sequence_in_cluster_array:", node_sequence_in_cluster_array)
    #print("node_included_in_cluster_array:", node_included_in_cluster_array)

    if (dpd_comm.dpd_debug_enable == 1):
        print("Target demand list", target_demand)
        print ("total demand of %d targets %d" % (no_of_target, sum(target_demand)))
        print("Number of cluster", no_of_cluster)
        print("node_sequence_in_cluster_array:", node_sequence_in_cluster_array) #array of k node sequences

    tsp_tours = []
    tour_distance_list = []

    for i in range(len(node_sequence_in_cluster_array)):
        #print("node sequence:", node_sequence_in_cluster_array[i])
        #christofides_algorithm
        if (len(node_sequence_in_cluster_array[i]) > 1):
            (tsp_tour, tour_distance) = chistofide_tsp(node_sequence_in_cluster_array[i])
        else:
            tour_distance = 0
            tsp_tour = [0, 0]
        tsp_tours.append(tsp_tour)
        tour_distance_list.append(tour_distance)
            

    #print("tsp_tours:", tsp_tours)
    return (node_sequence_in_cluster_array, tsp_tours, tour_distance_list)

#find the longest distance in a TSP given by path for nodes xx
def longest(xx, path):
    lo = 0.0
    f = 0.0
    l = 0.0
    xxs=xx[path]
    for i in range(len(path)-1):
        di = dpd_comm.distance(xxs[i][0], xxs[i+1][0], xxs[i][1], xxs[i+1][1])
        if (di>lo):
            f = xxs[i]
            l = xxs[i+1]
            lo = di
                 
    return(lo,f,l) #f and l are the nodes lo apart

def find_first_last_point_in_tsp_le(node_sequence_in_cluster_array, P):
    # F is the first point in each TSP
    # L is the last point in each TSP
    F = np.zeros((len(P), 2))
    L = np.zeros((len(P), 2))
    d = np.zeros((len(P)))
    for i in range(len(P)):
        if (len(P[i]) == 1):
            F[i] = P[i]
            L[i] = P[i]
            d[i] = 0
        else:
            (d[i], F[i], L[i]) = longest(node_sequence_in_cluster_array[i], P[i])
        
    return (F, L, d)

def find_drone_position(C, F, L):
    FL = np.concatenate((F,L), axis=0) #FL set of first, last points in each tsp
    dp =  dpd_comm.geometric_median(FL)
        
    return dp

#include the drone by removing longest edge of each tsp
def k_cap_drone_totalpath_traversal(XX, C, P, tour_distance_list, F, L, d, Drone): 
    total=0
    for i in range(len(P)):
        length = tour_distance_list[i] + dpd_comm.distance(Drone[0], F[i][0], Drone[1], F[i][1]) + dpd_comm.distance(Drone[0], L[i][0], Drone[1], L[i][1]) - d[i]
        #print (length) #length of each tsp with Drone included
        total = total + length

    return total

def k_cap_drone_total_traversal_longest_edge(target_loc_xy, no_of_target, drone_cap, loc, target_demand, no_of_cluster):
    #print("Number of targets", no_of_target)
    (centroids, x) = const_k_mean(loc, target_loc_xy, target_demand, no_of_cluster, drone_cap)

    (node_sequence_in_cluster_array, tsp_tours, tour_distance_list) = do_tsp(loc, no_of_cluster, target_loc_xy, target_demand, no_of_target, x)

    #print("P:", P)
    #print("node_sequence_in_cluster_array:", node_sequence_in_cluster_array)
    (F,L,d) = find_first_last_point_in_tsp_le(node_sequence_in_cluster_array, tsp_tours)
    #print("F:", F)
    #print("L:", L)
    dp = find_drone_position(centroids, F, L)
    #print("Drone:", Drone)

    """
    #Drone_path = find_path_with_drone(node_sequence_in_cluster_array, tsp_tours, F, L)
    #if (dpd_comm.dpd_debug_enable == 1):
    #    print("Path with Drone:", Drone_path)
    """

    #F: First target point of the TSP of that cluster
    #L: Last target point of the TSP of that cluster 
    total = k_cap_drone_totalpath_traversal(node_sequence_in_cluster_array, centroids, tsp_tours, tour_distance_list, F, L, d, dp)
    #print("total distance %d" % (total))

    return total

#GMTP related functions
def create_shifted_partitions(tour, k, shift):
    """
    tour: Christofides tour without repeated last node
    shift: starting offset
    """
    n = len(tour)

    shifted = []

    for i in range(n):
        shifted.append(tour[(shift + i) % n])

    partitions = []

    for i in range(0, n, k):
        partitions.append(shifted[i:i+k])

    return partitions

def partition_tour_length(dp, partition, target_loc_xy):

    if len(partition) == 0:
        return 0

    first = target_loc_xy[partition[0]]

    total = dpd_comm.distance(dp[0], first[0], dp[1], first[1])

    for i in range(len(partition)-1):
        p1 = target_loc_xy[partition[i]]
        p2 = target_loc_xy[partition[i+1]]

        total += dpd_comm.distance(p1[0], p2[0], p1[1], p2[1])

    last = target_loc_xy[partition[-1]]

    total += dpd_comm.distance(dp[0], last[0], dp[1], last[1])

    return total

def compute_partition_set_tfl(dp, partitions, target_loc_xy):

    total = 0

    for part in partitions:
        total += partition_tour_length(dp, part, target_loc_xy)

    return total

def k_cap_drone_total_traversal_gmtp(target_loc_xy, drone_cap):

    # Christofides Tour
    tsp_tour, _ = chistofide_tsp(target_loc_xy)

    # Remove duplicated last node
    if tsp_tour[0] == tsp_tour[-1]:
        tsp_tour = tsp_tour[:-1]

    n = len(tsp_tour)
    n_c = (int) (np.ceil(n/drone_cap))

    min_tfl = float("inf")
    best_dp = None

    # Evaluate all shifts
    for shift in range(drone_cap):

        partitions = create_shifted_partitions(tsp_tour, drone_cap, shift)

        boundary_nodes = []

        for part in partitions:

            first_node = target_loc_xy[part[0]]
            last_node = target_loc_xy[part[-1]]

            boundary_nodes.append(first_node)
            boundary_nodes.append(last_node)

        boundary_nodes = np.array(boundary_nodes)

        dp = dpd_comm.geometric_median(boundary_nodes)

        tfl = compute_partition_set_tfl(
            dp,
            partitions,
            target_loc_xy
        )

        if tfl < min_tfl:
            min_tfl = tfl

    return min_tfl


def ref_lower_bound_calculation(target_loc_xy, drone_cap):

    # Geometric median
    dp = dpd_comm.geometric_median(target_loc_xy)

    # Christofides tour
    christofides_tour, christofides_length = chistofide_tsp(target_loc_xy)

    # Average radial distance
    total_distance = dpd_sc.drone_totalpath_traversal_single_capacity(
        target_loc_xy, dp
    )

    r_bar = total_distance / len(target_loc_xy)

    # Haimovich & Rinnooy Kan lower bound components
    LB_TSP = (2.0 / 3.0) * christofides_length
    LB_Radial = (2.0 * r_bar * len(target_loc_xy)) / drone_cap

    #print(f"LB_TSP: {LB_TSP}, LB_Radial: {LB_Radial}")

    return max(LB_TSP, LB_Radial)

def k_cap_drone_total_traversal_gmtp_plot(target_loc_xy, drone_cap, debug=False):

    # Christofides Tour
    tsp_tour, _ = chistofide_tsp(target_loc_xy)

    # Remove duplicated last node
    if tsp_tour[0] == tsp_tour[-1]:
        tsp_tour = tsp_tour[:-1]

    min_tfl = float("inf")

    # Best solution
    best_dp = None
    best_shift = None
    best_partitions = None
    best_boundary_nodes = None

    # Evaluate all shifts
    for shift in range(drone_cap):

        partitions = create_shifted_partitions(
            tsp_tour,
            drone_cap,
            shift
        )

        boundary_nodes = []

        for part in partitions:

            boundary_nodes.append(target_loc_xy[part[0]])
            boundary_nodes.append(target_loc_xy[part[-1]])

        boundary_nodes = np.array(boundary_nodes)

        dp = dpd_comm.geometric_median(boundary_nodes)

        tfl = compute_partition_set_tfl(
            dp,
            partitions,
            target_loc_xy
        )

        if tfl < min_tfl:

            min_tfl = tfl

            best_dp = dp
            best_shift = shift
            best_partitions = [p.copy() for p in partitions]
            best_boundary_nodes = boundary_nodes.copy()

    if debug:

        return {
            "tfl": min_tfl,
            "dp": best_dp,
            "tour": tsp_tour,
            "shift": best_shift,
            "partitions": best_partitions,
            "boundary_nodes": best_boundary_nodes,
            "target_loc_xy": target_loc_xy
}

    return min_tfl