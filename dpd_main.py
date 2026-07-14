# importing the module
import numpy as np
import json
import dpd_single_capacity as dpd_sc
import dpd_double_capacity as dpd_dc
import dpd_k_capacity as dpd_kc
import pandas as pd

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

def plot_selected_customers(target_loc_xy,
                            no_of_target,
                            iteration,
                            save_pdf=False):
    """
    Plot the selected customer locations in the 5 km × 5 km region.
    """

    plt.figure(figsize=(6, 6))

    plt.scatter(target_loc_xy[:,0],
                target_loc_xy[:,1],
                s=18,
                marker='o')

    plt.xlim(0, 5000)
    plt.ylim(0, 5000)

    plt.xlabel("X Coordinate (m)")
    plt.ylabel("Y Coordinate (m)")
    plt.title(f"{no_of_target} Randomly Selected Customer Locations")

    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box')

    if save_pdf:
        plt.savefig(
            f"customer_locations_{no_of_target}_{iteration}.pdf",
            bbox_inches="tight"
        )

    plt.show()

# ----------------------------------------------------------
# Global dataset (loaded only once)
# ----------------------------------------------------------
delivery_df = None


def initialize_delivery_points_new(no_of_target,
                               W,
                               H,
                               sd,
                               csv_file="delivery_cq.csv"):

    global delivery_df

    np.random.seed(sd)

    #########################################################
    # Load the delivery dataset only once
    #########################################################

    if delivery_df is None:

        print("Loading delivery dataset...")

        delivery_df = pd.read_csv(csv_file)

        delivery_df = delivery_df[
            ['delivery_gps_lng', 'delivery_gps_lat']
        ].dropna()

        #####################################################
        # Convert GPS to Cartesian coordinates (meters)
        #####################################################

        lon = delivery_df['delivery_gps_lng'].to_numpy()
        lat = delivery_df['delivery_gps_lat'].to_numpy()

        lon0 = np.mean(lon)
        lat0 = np.mean(lat)

        R = 6371000.0

        x = R * np.radians(lon - lon0) * np.cos(np.radians(lat0))
        y = R * np.radians(lat - lat0)

        delivery_df['x'] = x
        delivery_df['y'] = y

        print("Loaded %d delivery locations." % len(delivery_df))

    #########################################################
    # Randomly search for a valid 5km x 5km window
    #########################################################

    WINDOW_SIZE = 5000.0          # meters
    HALF_WINDOW = WINDOW_SIZE / 2

    rng = np.random.default_rng(sd)

    while True:
        idx = rng.integers(len(delivery_df))
        
        #############################################
        # Pick a random customer as window center
        #############################################
        center = delivery_df.iloc[idx]

        cx = center['x']
        cy = center['y']

        #############################################
        # Extract all deliveries inside the window
        #############################################

        window = delivery_df[
            (delivery_df['x'] >= cx - HALF_WINDOW) &
            (delivery_df['x'] <= cx + HALF_WINDOW) &
            (delivery_df['y'] >= cy - HALF_WINDOW) &
            (delivery_df['y'] <= cy + HALF_WINDOW)
        ]

        #############################################
        # Check whether enough customers exist
        #############################################

        if len(window) < no_of_target:
            continue

        #############################################
        # Randomly sample required customers
        #############################################

        sample = window.sample(
            n=no_of_target,
            random_state=sd
        )

        #############################################
        # Convert to local coordinates
        #############################################

        sxy = sample[['x', 'y']].to_numpy()

        # Shift origin to lower-left corner
        sxy[:, 0] -= np.min(sxy[:, 0])
        sxy[:, 1] -= np.min(sxy[:, 1])

        break

    #########################################################
    # Unit demand
    #########################################################

    demand = np.ones(no_of_target, dtype=int)

    return sxy, demand

def load_config(file_path="dpd_input.txt"):
    """Loads the JSON configuration file."""
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        # Fallback default config if file is missing
        return {
            "X_Axis": 5000,
            "Y_Axis": 5000,
            "Drone_Capacity": [3, 5, 8],
            "Number_of_Target": [20, 40, 60],
            "Save_Result": 1
        }

#Randomly select n delivery points
def initialize_delivery_points(no_of_target, W, H, sd): #delivery(n,W,H,sd): #to take results varying n,sd
    np.random.seed(sd)
    demand=np.random.randint(1, 2, size=(no_of_target))
    ####### Generate points randomly but uniformly ########
    sx = np.random.randint(W, size=(no_of_target))
    sy = np.random.randint(H, size=(no_of_target))
    
    sxy = np.vstack((sx, sy)).T

    return(sxy, demand)

def lower_bound_calculation(positions, k):
    """
    Calculates the lower bound on Total Flight Length (TFL) using positions and drone capacity.
    
    Parameters:
        positions (list of tuple): A list of (x, y) coordinate tuples.
        k (int): Capacity of the drone.
    
    Returns:
        int or float: The computed lower bound (LB) on TFL.
    """
    # Extract x-coordinates and sort them
    P_x = sorted([p[0] for p in positions])
    LB_x = 0

    while len(P_x) >= 2 * k:
        x_min = min(P_x)
        # Remove k smallest elements
        del P_x[:k]
        x_max = max(P_x)
        # Remove k largest elements
        del P_x[-k:]
        LB_x += 2 * (x_max - x_min)

    if len(P_x) > 0:
        LB_x += 2 * (max(P_x) - min(P_x))

    # Extract y-coordinates and sort them
    P_y = sorted([p[1] for p in positions])
    LB_y = 0

    while len(P_y) >= 2 * k:
        y_min = min(P_y)
        # Remove k smallest elements
        del P_y[:k]
        y_max = max(P_y)
        # Remove k largest elements
        del P_y[-k:]
        LB_y += 2 * (y_max - y_min)

    if len(P_y) > 0:
        LB_y += 2 * (max(P_y) - min(P_y))

    LB = LB_x + LB_y

    LB = (LB) / np.sqrt(2)
    
    return LB


def main():
    print("****************** START ****************")

    config = load_config()

    print("Euclidean area  X=%d, Y=%d" % (config["X_Axis"], config["Y_Axis"]))
    results = []

    NO_OF_RUNS = 20

    save = 1

    for no_of_target in config["Number_of_Target"]:
        if (config["Verbose"] == 1):
            print(f"\nNumber of Targets : {no_of_target}")

        ####################################################
        # Lists common for all capacities
        ####################################################
        total_single_list = []
        total_double_list = []

        ####################################################
        # Separate lists for each capacity
        ####################################################
        le_list = {}
        gmtp_list = {}
        lb_list = {}
        ref_lb_list = {}

        for cap in config["Drone_Capacity"]:
            le_list[cap] = []
            gmtp_list[cap] = []
            lb_list[cap] = []
            ref_lb_list[cap] = []

        ####################################################
        # Repeat same experiment 20 times
        ####################################################
        for iteration in range(NO_OF_RUNS):
            if (config["Verbose"] == 1):
                print(f"Iteration : {iteration+1}")

            # Generate ONE random instance
            (target_loc_xy, target_demand) = initialize_delivery_points_new(no_of_target, config["X_Axis"], config["Y_Axis"], iteration)
            no_of_target = len(target_loc_xy)
            loc = np.arange(no_of_target)

            if (save == 2):
                plot_selected_customers(target_loc_xy, no_of_target, iteration, save_pdf=True)
                
            save = save+1

            # Run SCER once
            total_single = dpd_sc.single_cap_drone_total_traversal(target_loc_xy)
            total_single_list.append(total_single)

            # Run DCER once
            total_double = dpd_dc.double_cap_drone_total_traversal(target_loc_xy)
            total_double_list.append(total_double)

            # Run algorithms for every capacity
            for drone_cap in config["Drone_Capacity"]:
                if (config["Verbose"] == 1):
                    print(f"Capacity : {drone_cap}")

                #number of clusters
                no_of_cluster = int(np.ceil(sum(target_demand)/drone_cap))
                if (no_of_cluster > no_of_target):
                    no_of_cluster = no_of_target
                
                # Longest Edge
                total_le = dpd_kc.k_cap_drone_total_traversal_longest_edge(
                                                    target_loc_xy, 
                                                    no_of_target, 
                                                    drone_cap, 
                                                    loc, 
                                                    target_demand, 
                                                    no_of_cluster)
                le_list[drone_cap].append(total_le)

                # GMTP
                total_gmtp = dpd_kc.k_cap_drone_total_traversal_gmtp(target_loc_xy, drone_cap)
                gmtp_list[drone_cap].append(total_gmtp)

                # Proposed Lower Bound
                lower_bound = lower_bound_calculation(target_loc_xy, drone_cap)
                lb_list[drone_cap].append(lower_bound)
        
                # Reference Lower Bound
                ref_lower_bound = dpd_kc.ref_lower_bound_calculation(target_loc_xy, drone_cap)
                ref_lb_list[drone_cap].append(ref_lower_bound)

        # Compute averages
        avg_single = np.mean(total_single_list)
        avg_double = np.mean(total_double_list)

        for drone_cap in config["Drone_Capacity"]:
            avg_le = np.mean(le_list[drone_cap])
            avg_gmtp = np.mean(gmtp_list[drone_cap])
            avg_lb = np.mean(lb_list[drone_cap])
            avg_ref_lb = np.mean(ref_lb_list[drone_cap])

            results.append({
                "Targets": no_of_target,
                "Drone Capacity": drone_cap,
                "SCP": round(avg_single / 1000, 2),
                "DCP": round(avg_double / 1000, 2),
                "LE": round(avg_le / 1000, 2),
                "GMTP": round(avg_gmtp / 1000, 2),
                "LB": round(avg_lb / 1000, 2),
                "RLB": round(avg_ref_lb / 1000, 2)
            })

    # Save results
    if (config["Save_Result"] == 1):
        df = pd.DataFrame(results)
        df.to_excel("Drone_Final_Data.xlsx", index=False)
        print(df)

if __name__ == "__main__":
    main()

