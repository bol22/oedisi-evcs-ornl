"""
EV Parameters Module - Configurable EV Fleet Generation

This module generates EV fleet parameters (arrival times, departure times, initial SOC)
based on configuration from system.json or defaults. Supports per-station EV generation
for multi-bus EVCS deployments.

Configuration can be provided via:
1. system.json parameters (passed through static_inputs.json)
2. DEFAULT_CONFIG fallback 
"""

import numpy as np
import json
import os
from scipy.stats import truncnorm, lognorm

# ============================================================================
# DEFAULT CONFIGURATION
# ============================================================================
# These values are used if no configuration is provided via system.json.

DEFAULT_CONFIG = {
    # EV Fleet Configuration
    "num_evs_per_station": [15, 12, 13],  # EVs per EVCS station (total: 40)
    "max_charging_rate": 11,               # kW per charger
    "battery_capacity": 50,                # kWh per vehicle
    "charging_efficiency": 0.95,           # Energy efficiency
    "desired_soc": 1.0,                    # Target SOC (100%)

    # Initial SOC Distribution (Truncated Normal)
    "soc_mean": 0.3,                       # Mean initial SOC (30%)
    "soc_std": 0.2,                        # Standard deviation
    "soc_lower": 0.1,                      # Lower bound (10%)
    "soc_upper": 0.5,                      # Upper bound (50%)

    # Arrival Time Distribution (Truncated Normal, hours from midnight)
    # Default: Morning arrival for 24-hour simulation
    "arrival_mean": 9.0,                   # Mean arrival time (9 AM)
    "arrival_std": 1.225,                  # Standard deviation (sqrt(1.5))
    "arrival_lower": 7.0,                  # Earliest arrival (7 AM)
    "arrival_upper": 11.0,                 # Latest arrival (11 AM)

    # Departure Time Distribution (Shifted Lognormal)
    "departure_shift": 17.5,               # Shift parameter (5:30 PM)
    "departure_mu": 0.0,                   # Lognormal mu
    "departure_sigma": 0.9,                # Lognormal sigma

    # Time Parameters
    "total_hours": 24,                     # Simulation duration (hours)
    "control_interval": 0.25,              # Control interval (15 min = 0.25 hr)

    # Randomization
    "random_seed": 42,                    

    # Electricity price array (96 values for 24-hour simulation at 15-min intervals)
    "electricity_price": [
        0.05753, 0.03334, 0.03098, 0.02518, 0.0319, 0.03044, 0.02022, 0.02363, 0.02296, 0.02188,
        0.02401, 0.02413, 0.02406, 0.03101, 0.03044, 0.02183, 0.03054, 0.02495, 0.05245, 0.0323,
        0.02488, 0.03324, 0.03119, 0.03099, 0.03202, 0.03339, 0.10425, 0.06402, 0.07178, 0.06245,
        0.04043, 0.04192, 0.06465, 0.0374, 0.03107, 0.0294, 0.03059, 0.02309, 0.0318, 0.02344,
        0.03023, 0.03506, 0.03684, 0.03301, 0.03361, 0.03681, 0.03325, 0.02933, 0.03206, 0.03128,
        0.02815, 0.02668, 0.02626, 0.02622, 0.02683, 0.02419, 0.02501, 0.02475, 0.02456, 0.02561,
        0.02566, 0.0253, 0.02475, 0.02508, 0.02603, 0.02533, 0.02698, 0.02886, 0.02569, 0.03306,
        0.03671, 0.03396, 0.03308, 0.03461, 0.04199, 0.03632, 0.03482, 0.10161, 0.09881, 0.11398,
        0.11103, 0.05463, 0.04681, 0.05261, 0.07763, 0.03797, 0.04209, 0.03728, 0.04444, 0.06747,
        0.03306, 0.03282, 0.02967, 0.03829, 0.03522, 0.03454
    ]
}


def load_config(config_path="static_inputs.json"):
    """
    Load configuration from static_inputs.json if available.

    Args:
        config_path: Path to configuration file

    Returns:
        dict: Configuration dictionary, or empty dict if file not found
    """
    try:
        with open(config_path) as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def generate_ev_parameters(config=None):
    """
    Generate EV parameters based on configuration.

    This function creates all EV-related parameters including arrival times,
    departure times, initial SOC, and charging energy requirements. Parameters
    are generated per-station to support multi-bus EVCS deployments.

    Args:
        config: Configuration dictionary. If None, loads from static_inputs.json
                or uses DEFAULT_CONFIG.

    Returns:
        dict: Dictionary containing all EV parameters:
            - num_evs: Total number of EVs
            - num_evs_per_station: List of EVs per station
            - max_charging_rate: Maximum charging rate (kW)
            - battery_capacity: Battery capacity (kWh)
            - charging_efficiency: Charging efficiency
            - desired_soc: Target SOC
            - control_interval: Control interval (hours)
            - total_hours: Simulation duration (hours)
            - num_control_steps: Number of control steps
            - electricity_price: Price array
            - arrival_time_idx: Arrival time indices (flattened)
            - departure_time_idx: Departure time indices (flattened)
            - initial_soc: Initial SOC values (flattened)
            - charging_energy: Required charging energy (flattened)
            - arrival_time_idx_stations: List of arrival arrays per station
            - departure_time_idx_stations: List of departure arrays per station
            - initial_soc_stations: List of SOC arrays per station
            - charging_energy_stations: List of energy arrays per station
            - evcs_bus_assignment: Mapping of bus IDs to EV indices
    """
    # Load config from file if not provided
    if config is None:
        config = load_config()

    # Merge with defaults (config overrides defaults)
    cfg = {**DEFAULT_CONFIG, **config}

    # Set random seed for reproducibility
    np.random.seed(cfg.get("random_seed", 42))

    # Extract configuration values
    num_evs_per_station = cfg["num_evs_per_station"]
    num_evs = sum(num_evs_per_station)
    control_interval = cfg["control_interval"]
    total_hours = cfg["total_hours"]
    num_control_steps = int(total_hours / control_interval)

    # SOC distribution parameters
    soc_mean = cfg["soc_mean"]
    soc_std = cfg["soc_std"]
    soc_lower = cfg["soc_lower"]
    soc_upper = cfg["soc_upper"]

    # Arrival distribution parameters
    arr_mean = cfg["arrival_mean"]
    arr_std = cfg["arrival_std"]
    arr_lower = cfg["arrival_lower"]
    arr_upper = cfg["arrival_upper"]

    # Departure distribution parameters
    dep_shift = cfg["departure_shift"]
    dep_mu = cfg["departure_mu"]
    dep_sigma = cfg["departure_sigma"]

    # Battery parameters
    battery_capacity = cfg["battery_capacity"]
    desired_soc = cfg["desired_soc"]

    # Generate per-station EV parameters
    arrival_time_idx_stations = []
    departure_time_idx_stations = []
    initial_soc_stations = []
    charging_energy_stations = []

    ev_offset = 0
    evcs_bus = cfg.get("evcs_bus", ["48.1", "65.1", "76.1"])

    # Check if user provided evcs_bus_assignment
    user_assignment = cfg.get("evcs_bus_assignment")
    if user_assignment and len(user_assignment) > 0:
        # Use user-provided assignment
        evcs_bus_assignment = user_assignment
    else:
        # Auto-generate from evcs_bus and num_evs_per_station
        evcs_bus_assignment = {}
        for station_idx, n_evs in enumerate(num_evs_per_station):
            if station_idx < len(evcs_bus):
                bus_id = evcs_bus[station_idx]
                evcs_bus_assignment[bus_id] = list(range(ev_offset, ev_offset + n_evs))
            ev_offset += n_evs
        ev_offset = 0  # Reset for the loop below

    for station_idx, n_evs in enumerate(num_evs_per_station):
        # Generate arrival times (Truncated Normal)
        a_arr = (arr_lower - arr_mean) / arr_std
        b_arr = (arr_upper - arr_mean) / arr_std
        arrival_hr = truncnorm.rvs(a_arr, b_arr, loc=arr_mean, scale=arr_std, size=n_evs)
        arrival_idx = np.floor(arrival_hr / control_interval).astype(int)
        arrival_idx = np.clip(arrival_idx, 0, num_control_steps - 1)

        # Generate departure times (Shifted Lognormal with constraints)
        departure_hr = np.zeros(n_evs)
        departure_idx = np.zeros(n_evs, dtype=int)

        for i in range(n_evs):
            max_attempts = 1000
            attempts = 0
            while attempts < max_attempts:
                attempts += 1
                x = lognorm.rvs(s=dep_sigma, scale=np.exp(dep_mu))
                t_dep = x + dep_shift

                # Constraint: departure within simulation and after arrival
                if t_dep <= total_hours and t_dep > arrival_hr[i]:
                    dep_idx = int(np.floor(t_dep / control_interval))
                    if dep_idx > arrival_idx[i]:
                        departure_hr[i] = t_dep
                        departure_idx[i] = min(dep_idx, num_control_steps)
                        break

            # Fallback if no valid departure found
            if attempts >= max_attempts:
                departure_idx[i] = min(arrival_idx[i] + 4, num_control_steps)

        # Generate initial SOC (Truncated Normal)
        a_soc = (soc_lower - soc_mean) / soc_std
        b_soc = (soc_upper - soc_mean) / soc_std
        initial_soc_station = truncnorm.rvs(a_soc, b_soc, loc=soc_mean, scale=soc_std, size=n_evs)
        initial_soc_station = np.clip(initial_soc_station, soc_lower, soc_upper)

        # Calculate required charging energy
        charging_energy_station = (desired_soc - initial_soc_station) * battery_capacity

        # Store per-station arrays
        arrival_time_idx_stations.append(arrival_idx)
        departure_time_idx_stations.append(departure_idx)
        initial_soc_stations.append(initial_soc_station)
        charging_energy_stations.append(charging_energy_station)

        ev_offset += n_evs

    # Create electricity price array
    electricity_price = np.array(cfg["electricity_price"])

    # Return parameters dictionary
    return {
        # Scalar parameters
        "num_evs": num_evs,
        "num_evs_per_station": num_evs_per_station,
        "max_charging_rate": cfg["max_charging_rate"],
        "battery_capacity": battery_capacity,
        "charging_efficiency": cfg["charging_efficiency"],
        "desired_soc": desired_soc,
        "control_interval": control_interval,
        "total_hours": total_hours,
        "num_control_steps": num_control_steps,

        # Price array
        "electricity_price": electricity_price,

        # Flattened arrays (for backward compatibility)
        "arrival_time_idx": np.concatenate(arrival_time_idx_stations),
        "departure_time_idx": np.concatenate(departure_time_idx_stations),
        "initial_soc": np.concatenate(initial_soc_stations),
        "charging_energy": np.concatenate(charging_energy_stations),

        # Per-station arrays (for multi-bus support)
        "arrival_time_idx_stations": arrival_time_idx_stations,
        "departure_time_idx_stations": departure_time_idx_stations,
        "initial_soc_stations": initial_soc_stations,
        "charging_energy_stations": charging_energy_stations,

        # Bus assignment
        "evcs_bus_assignment": evcs_bus_assignment,
    }


# ============================================================================
# BACKWARD COMPATIBILITY: Global Variables
# ============================================================================
# These global variables maintain backward compatibility 
# that imports parameters directly (e.g., from ev_parameters import num_evs).
# They are generated using DEFAULT_CONFIG at module load time.

# Generate default parameters
_default_params = generate_ev_parameters(DEFAULT_CONFIG)

# Export as global variables
num_evs = _default_params["num_evs"]
max_charging_rate = _default_params["max_charging_rate"]
charging_efficiency = _default_params["charging_efficiency"]
battery_capacity = _default_params["battery_capacity"]
control_interval = _default_params["control_interval"]
total_hours = _default_params["total_hours"]
num_control_steps = _default_params["num_control_steps"]
electricity_price = _default_params["electricity_price"]
arrival_time_idx = _default_params["arrival_time_idx"]
departure_time_idx = _default_params["departure_time_idx"]
initial_soc = _default_params["initial_soc"]
charging_energy = _default_params["charging_energy"]
desired_state_of_charge = _default_params["desired_soc"]
evcs_bus_assignment = _default_params["evcs_bus_assignment"]

# Legacy variables for compatibility 
arrival_time_hr = None  # Not exposed as global (use arrival_time_idx)
departure_time_hr = None  # Not exposed as global (use departure_time_idx)
t0_dep = DEFAULT_CONFIG["departure_shift"]
mu_a_dep = DEFAULT_CONFIG["departure_mu"]
sigma_a_dep = DEFAULT_CONFIG["departure_sigma"]
dep_upper_bound_hr = DEFAULT_CONFIG["total_hours"]
soc_ini_mean = DEFAULT_CONFIG["soc_mean"]
soc_ini_std_dev = DEFAULT_CONFIG["soc_std"]
soc_ini_lower_bound = DEFAULT_CONFIG["soc_lower"]
soc_ini_upper_bound = DEFAULT_CONFIG["soc_upper"]


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_bus_for_ev(ev_index, bus_assignment=None):
    """
    Return the bus ID where this EV is located.

    Args:
        ev_index: Index of the EV
        bus_assignment: Optional bus assignment dict. Uses global if None.

    Returns:
        str: Bus ID (e.g., "48.1")
    """
    if bus_assignment is None:
        bus_assignment = evcs_bus_assignment

    for bus, ev_list in bus_assignment.items():
        if ev_index in ev_list:
            return bus
    # Default to first bus if not found
    return list(bus_assignment.keys())[0]


def get_evs_at_bus(bus_id, bus_assignment=None):
    """
    Return list of EV indices assigned to a specific bus.

    Args:
        bus_id: Bus ID string (e.g., "48.1")
        bus_assignment: Optional bus assignment dict. Uses global if None.

    Returns:
        list: List of EV indices at this bus
    """
    if bus_assignment is None:
        bus_assignment = evcs_bus_assignment

    return bus_assignment.get(bus_id, [])


def update_global_parameters(config):
    """
    Update global parameters based on new configuration.

    This function regenerates all EV parameters and updates the global
    variables. Use this when configuration changes at runtime.

    Args:
        config: New configuration dictionary
    """
    global num_evs, max_charging_rate, charging_efficiency, battery_capacity
    global control_interval, total_hours, num_control_steps, electricity_price
    global arrival_time_idx, departure_time_idx, initial_soc, charging_energy
    global desired_state_of_charge, evcs_bus_assignment

    params = generate_ev_parameters(config)

    num_evs = params["num_evs"]
    max_charging_rate = params["max_charging_rate"]
    charging_efficiency = params["charging_efficiency"]
    battery_capacity = params["battery_capacity"]
    control_interval = params["control_interval"]
    total_hours = params["total_hours"]
    num_control_steps = params["num_control_steps"]
    electricity_price = params["electricity_price"]
    arrival_time_idx = params["arrival_time_idx"]
    departure_time_idx = params["departure_time_idx"]
    initial_soc = params["initial_soc"]
    charging_energy = params["charging_energy"]
    desired_state_of_charge = params["desired_soc"]
    evcs_bus_assignment = params["evcs_bus_assignment"]
