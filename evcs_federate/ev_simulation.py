import numpy as np
import opendssdirect as dss
import os

def run_opendss_simulation(total_ev_load_kw, feeder_loads_p, feeder_loads_q):
    """
    Runs OpenDSS power flow with EV load and feeder loads.
    Returns per-unit voltages at all buses for constraint checking.

    The EVCS federate has its own local OpenDSS model (master.dss) for
    internal voltage constraint checking during PSO optimization.
    """
    # Load bus IDs (from IEEE 123-bus model)
    load_ids = ['1.1', '2.2', '4.3', '5.3', '6.3', '7.1', '9.1', '10.1', '11.1', '12.2',
                '16.3', '17.3', '19.1', '20.1', '22.2', '24.3', '28.1', '29.1', '30.3',
                '31.3', '32.3', '33.1', '34.3', '35.1', '35.2', '37.1', '38.2', '39.2',
                '41.3', '42.1', '43.2', '45.1', '46.1', '47.1', '47.2', '47.3', '48.1',
                '48.2', '48.3', '49.1', '49.2', '49.3', '50.3', '51.1', '52.1', '53.1',
                '55.1', '56.2', '58.2', '59.2', '60.1', '62.3', '63.1', '64.2', '65.1',
                '65.2', '65.3', '66.3', '68.1', '69.1', '70.1', '71.1', '73.3', '74.3',
                '75.3', '76.1', '76.2', '76.3', '77.2', '79.1', '80.2', '82.1', '83.3',
                '84.3', '85.3', '86.2', '87.2', '88.1', '90.2', '92.3', '94.1', '95.2',
                '96.2', '98.1', '99.2', '100.3', '102.3', '103.3', '104.3', '106.2',
                '107.2', '109.1', '111.1', '112.1', '113.1', '114.1']

    from ev_parameters import evcs_bus

    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    master_dss_path = os.path.join(script_dir, 'master.dss')

    # Compile local OpenDSS model
    dss.Text.Command("clear")
    dss.Text.Command(f'Compile ({master_dss_path})')

    # Set load values from feeder data
    all_loads = dss.Loads.AllNames()
    for load_name in all_loads:
        # Try to find matching load in feeder data
        for load_id in load_ids:
            bus_name = load_id.split('.')[0]
            if load_name.lower() == bus_name.lower() or load_name.lower() == f's{bus_name}'.lower():
                if load_id in feeder_loads_p:
                    try:
                        dss.Loads.Name(load_name)
                        dss.Loads.kW(feeder_loads_p[load_id])
                        if load_id in feeder_loads_q:
                            dss.Loads.kvar(feeder_loads_q[load_id])
                    except Exception:
                        pass
                break

    # Add EV load at EVCS bus
    for bus in evcs_bus:
        bus_name = bus.split('.')[0]
        ev_load_name = f"EV_{bus_name}"

        # Check if EV load already exists
        if ev_load_name.lower() not in [n.lower() for n in dss.Loads.AllNames()]:
            # Create new EV load
            dss.Text.Command(f"New Load.{ev_load_name} Bus1={bus} kW={total_ev_load_kw} kvar=0 model=1")
        else:
            # Update existing EV load
            dss.Loads.Name(ev_load_name)
            dss.Loads.kW(total_ev_load_kw)

    # Solve power flow
    dss.Text.Command("Set mode = snapshot")
    dss.Text.Command("Solve")

    if not dss.Solution.Converged():
        # Return high voltages to trigger penalty if not converged
        return np.ones(len(load_ids)) * 1.1

    # Get voltage magnitudes
    all_node_names = [n.lower() for n in dss.Circuit.AllNodeNames()]
    all_volt_mag = np.array(dss.Circuit.AllBusMagPu())

    # Extract voltages at load buses
    voltages = []
    for load_id in load_ids:
        if load_id.lower() in all_node_names:
            idx = all_node_names.index(load_id.lower())
            voltages.append(all_volt_mag[idx])
        else:
            # If bus not found, assume nominal voltage
            voltages.append(1.0)

    return np.array(voltages)

def uncontrolled_charging(initial_soc, num_control_steps, control_interval, battery_capacity, charging_efficiency, arrival_time_idx, departure_time_idx, num_evs, max_charging_rate, desired_state_of_charge):
    """
    Simulates uncontrolled charging: Charge at max rate until target SOC is reached.
    """
    soc = np.zeros((num_evs, num_control_steps))
    charging_rate = np.zeros((num_evs, num_control_steps))
    soc_tolerance = 0.001 # Tolerance for checking if target SOC is reached

    # Set initial SOC at the arrival time step for each EV
    for ev in range(num_evs):
        if arrival_time_idx[ev] < num_control_steps:
             soc[ev, arrival_time_idx[ev]] = initial_soc[ev]
             if arrival_time_idx[ev] > 0:
                 soc[ev, 0:arrival_time_idx[ev]] = initial_soc[ev] # Propagate backwards

    # Simulate step-by-step
    for t in range(num_control_steps - 1):
        for ev in range(num_evs):
            # Check if EV is present and needs charging
            is_present = (t >= arrival_time_idx[ev] and t < departure_time_idx[ev])
            needs_charge = (soc[ev, t] < desired_state_of_charge - soc_tolerance)

            if is_present and needs_charge:
                # Charge at maximum rate
                charging_rate[ev, t] = max_charging_rate
            else:
                # No charging if not present or already charged
                charging_rate[ev, t] = 0

            # Calculate SOC for the next step based on the decided charging rate
            if is_present:
                charged_energy = charging_rate[ev, t] * control_interval * charging_efficiency
                soc[ev, t + 1] = soc[ev, t] + (charged_energy / battery_capacity)
            else:
                # SOC remains the same if EV is not present
                soc[ev, t + 1] = soc[ev, t]

            # Clip SOC to physical limits [0, 1]
            soc[ev, t + 1] = np.clip(soc[ev, t + 1], 0, desired_state_of_charge)

    # Ensure charging rate is 0 for the last step (as SOC is calculated for t+1)
    charging_rate[:, -1] = 0

    return soc, charging_rate


def calculate_soc(initial_soc, charging_rate, num_control_steps, control_interval, 
                  battery_capacity, charging_efficiency, arrival_time_idx, departure_time_idx, num_evs):
    """
    Calculate the State of Charge (SOC) for each EV over time steps.
    Args are updated to use indices and control interval.
    """
    soc = np.zeros((num_evs, num_control_steps))
    # Set initial SOC at the arrival time step for each EV
    for ev in range(num_evs):
        if arrival_time_idx[ev] < num_control_steps:
             soc[ev, arrival_time_idx[ev]] = initial_soc[ev]
             # Propagate initial SOC backwards if arrival is not 0 (though less physically intuitive, needed for array structure)
             if arrival_time_idx[ev] > 0:
                 soc[ev, 0:arrival_time_idx[ev]] = initial_soc[ev]


    for t in range(num_control_steps - 1):
        for ev in range(num_evs):
            # Only charge if the EV is present at the charging station during this interval
            if t >= arrival_time_idx[ev] and t < departure_time_idx[ev]:
              # Energy charged in this time interval (Power * time * efficiency)
              charged_energy = charging_rate[ev, t] * control_interval * charging_efficiency
              soc[ev, t + 1] = soc[ev, t] + (charged_energy / battery_capacity)
            else:
               # If EV not present or already departed, SOC remains the same as the previous step
               soc[ev, t + 1] = soc[ev, t]
            # Clip SOC to physical limits [0, 1]
            soc[ev, t + 1] = np.clip(soc[ev, t + 1], 0, 1)
    return soc




# def calculate_soc_and_adjust_rates(initial_soc,charging_rate,desired_soc,num_control_steps,control_interval,
#                                    battery_capacity,charging_efficiency,arrival_time_idx, departure_time_idx, num_evs):
#     """
#     Calculates the State of Charge (SOC) and returns a revised charging schedule.

#     This improved logic stops charging an EV for all subsequent time steps
#     once its desired_soc is reached.

#     Args:
#         initial_soc (np.array): The starting SOC for each EV.
#         charging_rate (np.array): The proposed charging schedule from the optimizer.
#         desired_soc (np.array): The target SOC for each EV. Can be less than 1.0.
#         All other parameters are the same as before.

#     Returns:
#         tuple: A tuple containing:
#             - soc (np.array): The resulting (30, 96) SOC trajectory table.
#             - revised_charging_rate (np.array): The adjusted (30, 96) charging rate
#               schedule where rates are set to 0 after the target SOC is met.
#     """
#     # Create a copy of the charging_rate to modify it without side effects
#     revised_charging_rate = charging_rate.copy()
    
#     # Initialize the SOC table
#     soc = np.zeros((num_evs, num_control_steps))

#     # Set initial SOC for all EVs, propagating backwards for array structure
#     for ev in range(num_evs):
#         if arrival_time_idx[ev] < num_control_steps:
#             soc[ev, :arrival_time_idx[ev] + 1] = initial_soc[ev]

#     # --- Main Simulation Loop ---
#     # We loop through each EV individually to handle its unique stop-charging logic.
#     for ev in range(num_evs):
#         for t in range(num_control_steps - 1):
            
#             # Condition 1: Is the EV present at the station?
#             is_present = (t >= arrival_time_idx[ev] and t < departure_time_idx[ev])
            
#             if is_present:
#                 # Condition 2: Has the EV already reached its desired SOC?
#                 if soc[ev, t] >= desired_soc:
#                     # If target is met, stop charging for this and all future steps
#                     revised_charging_rate[ev, t:] = 0
#                     # SOC carries over, no change
#                     soc[ev, t + 1] = soc[ev, t]
#                 else:
#                     # EV is present and needs charging.
#                     # Use the (potentially revised) charging rate for the current step
#                     current_rate = revised_charging_rate[ev, t]
                    
#                     charged_energy = current_rate * control_interval * charging_efficiency
#                     soc_gain = charged_energy / battery_capacity # Assuming battery_capacity is also an array
                    
#                     soc[ev, t + 1] = soc[ev, t] + soc_gain
#             else:
#                 # If EV is not present, SOC carries over
#                 soc[ev, t + 1] = soc[ev, t]
            
#             # Finally, clip the result to ensure it's within physical limits [0, 1]
#             soc[ev, t + 1] = np.clip(soc[ev, t + 1], 0, 1)

#     return soc, revised_charging_rate



# --- Calculate Results for Uncontrolled Scenario ---
# Copy helper functions from PSO script for consistency
def calculate_cost(charging_rate, electricity_price, num_control_steps, control_interval, num_evs, arrival_time_idx, departure_time_idx):
    """Calculate the total charging cost."""
    total_cost = 0
    for t in range(num_control_steps):
        for ev in range(num_evs):
             if t >= arrival_time_idx[ev] and t < departure_time_idx[ev]:
                total_cost += charging_rate[ev, t] * control_interval * electricity_price[t]
    return total_cost

def calculate_cost_per_step(charging_rate, electricity_price, num_control_steps, control_interval, num_evs, arrival_time_idx, departure_time_idx):
    """Calculate the charging cost incurred at each time step across all EVs."""
    cost_per_step = np.zeros(num_control_steps)
    for t in range(num_control_steps):
        step_cost = 0
        for ev in range(num_evs):
             if t >= arrival_time_idx[ev] and t < departure_time_idx[ev]:
                step_cost += charging_rate[ev, t] * control_interval * electricity_price[t]
        cost_per_step[t] = step_cost
    return cost_per_step


def fitness_function(charging_rate, feeder_loads_p, feeder_loads_q):
    """
    Fitness function for PSO. Minimize cost while meeting SOC and grid voltage requirements.
    """
    
    from ev_parameters import (
        num_evs, max_charging_rate, charging_efficiency, battery_capacity,
        control_interval, total_hours, num_control_steps,
        electricity_price, arrival_time_hr, arrival_time_idx,
        t0_dep, mu_a_dep, sigma_a_dep, dep_upper_bound_hr,
        departure_time_hr, departure_time_idx,
        soc_ini_mean, soc_ini_std_dev, soc_ini_lower_bound, soc_ini_upper_bound,
        initial_soc, desired_state_of_charge, charging_energy
    )
  
    # Pass necessary parameters to helper functions
    soc = calculate_soc(initial_soc, charging_rate, num_control_steps, control_interval, battery_capacity, charging_efficiency, arrival_time_idx, departure_time_idx, num_evs)
    cost = calculate_cost(charging_rate, electricity_price, num_control_steps, control_interval, num_evs, arrival_time_idx, departure_time_idx)
    
    # --- Penalties ---
    undershoot_penalty = 0
    voltage_penalty = 0

    # Penalty for not meeting the required SoC
    for ev in range(num_evs):
        dep_check_idx = departure_time_idx[ev]
        if dep_check_idx >= 0 and dep_check_idx < num_control_steps:
            final_soc = soc[ev, dep_check_idx]
            if final_soc < desired_state_of_charge:
              undershoot_penalty += 100

    # Penalty for voltage violations (only if OpenDSS is available)
    total_ev_load_kw = np.sum(charging_rate, axis=0)
    for t in range(num_control_steps):
        voltages = run_opendss_simulation(total_ev_load_kw[t], feeder_loads_p, feeder_loads_q)
        # Skip voltage checking if OpenDSS is not available (co-simulation mode)
        if voltages is not None:
            if np.any(voltages > 1.05) or np.any(voltages < 0.95):
                voltage_penalty += 1000

    # Total fitness is cost plus all penalties
    return cost + undershoot_penalty + voltage_penalty
   
     


def ev_pso_optimization(num_particles, max_iterations, feeder_loads_p, feeder_loads_q):
    """
    Performs Particle Swarm Optimization to find the optimal charging schedule for electric vehicles.

    Args:
        num_particles (int): The number of particles in the swarm.
        max_iterations (int): The maximum number of iterations for the optimization.
        num_evs (int): The number of electric vehicles.
        max_charging_rate (float): The maximum charging rate for an EV.
        arrival_time_idx (list): A list of arrival time indices for each EV.
        departure_time_idx (list): A list of departure time indices for each EV.
        num_control_steps (int): The number of control steps in the optimization horizon.
        fitness_function (function): The function to evaluate the fitness of a particle's position.
        inertia_weight (float): The inertia weight for the PSO velocity update.
        cognitive_coefficient (float): The cognitive coefficient for the PSO velocity update.
        social_coefficient (float): The social coefficient for the PSO velocity update.

    Returns:
        tuple: A tuple containing the global best position and the global best fitness.
    """
    
    # Import EV parameters
    from ev_parameters import (
        num_evs, max_charging_rate, charging_efficiency, battery_capacity,
        control_interval, total_hours, num_control_steps,
        electricity_price, arrival_time_hr, arrival_time_idx,
        t0_dep, mu_a_dep, sigma_a_dep, dep_upper_bound_hr,
        departure_time_hr, departure_time_idx,
        soc_ini_mean, soc_ini_std_dev, soc_ini_lower_bound, soc_ini_upper_bound,
        initial_soc, desired_state_of_charge, charging_energy
    )
    # Initialize particles and velocities   
    
    particles = np.random.uniform(0, max_charging_rate, (num_particles, num_evs, num_control_steps))

    # Ensure initial particles respect arrival/departure times
    for p in range(num_particles):
        for ev in range(num_evs):
            particles[p, ev, :arrival_time_idx[ev]] = 0
            particles[p, ev, departure_time_idx[ev]:] = 0
            
            # # Set the remaining items in particles to zero to avoid overcharging
            # cumulated_energy = 0
            # for j in range(num_control_steps):
            #     cumulated_energy = cumulated_energy + particles[p, ev, j] * control_interval
            #     if cumulated_energy > charging_energy[ev]:
            #         particles[p, ev, j+1:] = 0
            #         break
               

    velocities = np.random.uniform(-max_charging_rate * 0.1, max_charging_rate * 0.1, (num_particles, num_evs, num_control_steps))
    personal_best_positions = particles.copy()
    personal_best_fitnesses = np.array([fitness_function(particles[i], feeder_loads_p, feeder_loads_q) for i in range(num_particles)])
    global_best_index = np.argmin(personal_best_fitnesses)
    global_best_position = personal_best_positions[global_best_index].copy()
    global_best_fitness = personal_best_fitnesses[global_best_index]

    print(f"Initial Best Fitness: {global_best_fitness:.2f}")

    # PSO optimization loop
    
    # PSO parameters
    inertia_weight = 0.7
    cognitive_coefficient = 1.4
    social_coefficient = 1.4
    
    for iteration in range(max_iterations):
        for i in range(num_particles):
            # Update velocity
            r1, r2 = np.random.rand(2)  # Random numbers for cognitive and social components
            velocities[i] = (inertia_weight * velocities[i] +
                             cognitive_coefficient * r1 * (personal_best_positions[i] - particles[i]) +
                             social_coefficient * r2 * (global_best_position - particles[i]))

            # Limit velocity to prevent explosion
            velocities[i] = np.clip(velocities[i], -max_charging_rate * 0.1, max_charging_rate * 0.1)

            # Update particle position
            particles[i] = particles[i] + velocities[i]

            # Apply constraints: charging rate limits and only charge when EV is present
            for ev in range(num_evs):
                # Set charging rate to 0 if EV is not present
                particles[i, ev, :arrival_time_idx[ev]] = 0
                particles[i, ev, departure_time_idx[ev]:] = 0
                # Clip charging rate within the allowed time window
                particles[i, ev, arrival_time_idx[ev]:departure_time_idx[ev]] = np.clip(
                    particles[i, ev, arrival_time_idx[ev]:departure_time_idx[ev]], 0, max_charging_rate
                )
                
                # Set the remaining items in particles to zero to avoid overcharging
                cumulated_energy = 0
                for j in range(num_control_steps):
                    cumulated_energy = cumulated_energy + particles[i, ev, j] * control_interval
                    if cumulated_energy > 1.05 * charging_energy[ev]:
                        particles[i, ev, j+1:] = 0
                        break
                
            # Evaluate fitness
            current_fitness = fitness_function(particles[i], feeder_loads_p, feeder_loads_q)

            # Update personal best
            if current_fitness < personal_best_fitnesses[i]:
                personal_best_positions[i] = particles[i].copy()
                personal_best_fitnesses[i] = current_fitness

                # Update global best if this particle is now the best
                if current_fitness < global_best_fitness:
                    global_best_position = particles[i].copy()
                    global_best_fitness = current_fitness

        print(f"Iteration {iteration + 1}/{max_iterations}: Best Fitness = {global_best_fitness:.2f}")

    return global_best_position, global_best_fitness
