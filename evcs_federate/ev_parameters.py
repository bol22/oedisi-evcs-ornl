import numpy as np
# import matplotlib.pyplot as plt
from scipy.stats import truncnorm, lognorm # Import necessary distributions
# import matplotlib.pyplot as plt
# import matplotlib.ticker as ticker # Import ticker for locator control
from scipy.stats import truncnorm, lognorm # Import necessary distributions
np.random.seed(40)

#%% Uncontrolled
num_evs = 40  # Number of electric vehicles in EVCS
max_charging_rate = 11  # Maximum charging rate (kW)
charging_efficiency = 0.95  # Efficiency of the charging process
battery_capacity = 50 #kWh. Assume all EVs have same capacity for simplicity

# Time parameters
control_interval = 0.25 # 15 minutes in hours
total_hours = 24
num_control_steps = int(total_hours / control_interval) # Should be 96

# Electricity price: Repeat hourly prices for each 15-min interval

electricity_price = np.array([0.05753, 0.03334, 0.03098, 0.02518, 0.0319, 0.03044, 0.02022, 0.02363, 0.02296, 0.02188,
    0.02401, 0.02413, 0.02406, 0.03101, 0.03044, 0.02183, 0.03054, 0.02495, 0.05245, 0.0323,
    0.02488, 0.03324, 0.03119, 0.03099, 0.03202, 0.03339, 0.10425, 0.06402, 0.07178, 0.06245,
    0.04043, 0.04192, 0.06465, 0.0374, 0.03107, 0.0294, 0.03059, 0.02309, 0.0318, 0.02344,
    0.03023, 0.03506, 0.03684, 0.03301, 0.03361, 0.03681, 0.03325, 0.02933, 0.03206, 0.03128,
    0.02815, 0.02668, 0.02626, 0.02622, 0.02683, 0.02419, 0.02501, 0.02475, 0.02456, 0.02561,
    0.02566, 0.0253, 0.02475, 0.02508, 0.02603, 0.02533, 0.02698, 0.02886, 0.02569, 0.03306,
    0.03671, 0.03396, 0.03308, 0.03461, 0.04199, 0.03632, 0.03482, 0.10161, 0.09881, 0.11398,
    0.11103, 0.05463, 0.04681, 0.05261, 0.07763, 0.03797, 0.04209, 0.03728, 0.04444, 0.06747,
    0.03306, 0.03282, 0.02967, 0.03829, 0.03522, 0.03454])

# electricity_price = np.array([
#     44.9, 126.11, 42.39, 57.9, 55.88, 50.98, 45.97, 45.92, 48.73, 54.91,
#     46.69, 46.48, 44.67, 51.78, 58.45, 53.95, 48.71, 44.27, 44.32, 41.46,
#     42.24, 38.18, 44.36, 44.41, 45.5, 45.12, 54, 72.5, 53.01, 45.07, 47.86,
#     98.37, 76.6, 54.27, 53.73, 74.05, 73.94, 73.89, 76.53, 74.11, 107.9,
#     73.37, 87.9, 88.86, 86.6, 76.65, 94.38, 99.85, 99.85, 100.48, 74.16,
#     70.24, 76.85, 73.93, 76.9, 88.21, 93.53, 72.54, 76.78, 129.31, 170.06,
#     84.44, 75.28, 71.56, 71.58, 55.11, 55.15, 77.12, 116.25, 196.27, 215.58,
#     224.13, 267.57, 100.2, 74.66, 74.93, 74.81, 90.8, 88.9, 88.33, 75.22,
#     63.03, 68.47, 60.04, 76.35, 59.8, 60.43, 60.58, 70.26, 75.38, 72.49,
#     48.76, 52.09, 54.89, 49.02, 37.45])  # $/MWh
# electricity_price = electricity_price/1000 # $/MWh to $/kWh


# Arrival Time (t_arr): Truncated Normal (hours from midnight)
arr_mean = 9.0
arr_std_dev = np.sqrt(1.5)
arr_lower_bound_hr = 7.0
arr_upper_bound_hr = 11.0

# Parameters for truncnorm (standardized bounds)
a_arr = (arr_lower_bound_hr - arr_mean) / arr_std_dev
b_arr = (arr_upper_bound_hr - arr_mean) / arr_std_dev
# Sample arrival times in hours
arrival_time_hr = truncnorm.rvs(a_arr, b_arr, loc=arr_mean, scale=arr_std_dev, size=num_evs)
# Convert arrival time in hours to control step index
arrival_time_idx = np.floor(arrival_time_hr / control_interval).astype(int)
# Ensure index is within bounds [0, num_control_steps - 1]
arrival_time_idx = np.clip(arrival_time_idx, 0, num_control_steps - 1)


# Departure Time (t_dep): 3-Parameter Lognormal (Shifted Lognormal)
t0_dep = 17.5 # Shift parameter (hours)
mu_a_dep = 0.0 # Mean of the underlying normal distribution
sigma_a_dep = 0.9 # Standard deviation of the underlying normal distribution
dep_upper_bound_hr = 24.0 # Constraint t <= 24 (hours)
departure_time_hr = np.zeros(num_evs)
departure_time_idx = np.zeros(num_evs, dtype=int)

for i in range(num_evs):
    while True:

        x_sample = lognorm.rvs(s=sigma_a_dep, scale=np.exp(mu_a_dep), size=1)
        t_dep_hr_candidate = x_sample[0] + t0_dep

        if t_dep_hr_candidate <= dep_upper_bound_hr and t_dep_hr_candidate > arrival_time_hr[i]:
            departure_time_hr[i] = t_dep_hr_candidate
            # Convert departure time in hours to control step index
            # Use floor for departure, ensuring the EV is gone *before* the start of the next interval if it departs exactly on the boundary
            dep_idx_candidate = np.floor(departure_time_hr[i] / control_interval).astype(int)
            # Ensure departure index is strictly after arrival index
            if dep_idx_candidate > arrival_time_idx[i]:
                 departure_time_idx[i] = np.clip(dep_idx_candidate, arrival_time_idx[i] + 1, num_control_steps) # Clip to be within bounds and after arrival
                 break # Valid departure time found
            # If dep_idx_candidate <= arrival_time_idx[i], loop continues to resample


# Initial SoC (SoC_ini): Truncated Normal (fraction)
soc_ini_mean = 0.3
soc_ini_std_dev = 0.2
soc_ini_lower_bound = 0.1
soc_ini_upper_bound = 0.5
# Parameters for truncnorm (standardized bounds)
a_soc = (soc_ini_lower_bound - soc_ini_mean) / soc_ini_std_dev
b_soc = (soc_ini_upper_bound - soc_ini_mean) / soc_ini_std_dev
# Sample initial SoC
initial_soc = truncnorm.rvs(a_soc, b_soc, loc=soc_ini_mean, scale=soc_ini_std_dev, size=num_evs)
initial_soc = np.clip(initial_soc, soc_ini_lower_bound, soc_ini_upper_bound) # Ensure strict bounds


# Required SoC (SoC_req): Fixed (fraction)
desired_state_of_charge = 1.0 # Target SOC

charging_energy = (desired_state_of_charge - initial_soc) * battery_capacity


# EV charging station 
evcs_bus = ['48.1']


