# importing the module
import dpd_common as dpd_comm
import cvxpy as cp
from sympy import *

#include the drone by removing longest edge of each tsp
def drone_totalpath_traversal_single_capacity(loc, Drone): 
    total=0
    for i in range(len(loc)):
        length = dpd_comm.distance(Drone[0], loc[i][0], Drone[1], loc[i][1])
        #print (length) #length of each tsp with Drone included
        total = total + length

    return total

def single_cap_drone_total_traversal(target_loc_xy):
    
    (drone_position) = dpd_comm.geometric_median(target_loc_xy)
    total = drone_totalpath_traversal_single_capacity(target_loc_xy, drone_position)
    #print("Drone position:", drone_position)
    #print("total distance %d" % (2*EC))
    #print("total distance %d" % (2*total))

    total = 2 * total

    return total
        