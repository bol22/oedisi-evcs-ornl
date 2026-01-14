"""
EV Simulation Module - PSO Optimization 
"""

import numpy as np
import opendssdirect as dss
import os


def run_opendss_simulation(ev_loads_per_bus, feeder_loads_p, feeder_loads_q, load_ids, evcs_bus):
    """
    Runs OpenDSS power flow with EV load and feeder loads.
    Returns per-unit voltages at all buses for constraint checking.

    The EVCS federate has its own local OpenDSS model (master.dss) for
    internal voltage constraint checking during PSO optimization.

    Args:
        ev_loads_per_bus: Dict mapping bus_id to EV load in kW
                          e.g., {"48.1": 45.3, "65.1": 32.1, "76.1": 28.5}
                          For backward compatibility, also accepts a scalar (total load)
        feeder_loads_p: Dict of bus_id -> real power from feeder
        feeder_loads_q: Dict of bus_id -> reactive power from feeder
        load_ids: List of load bus IDs (from PowersReal.ids)
        evcs_bus: List of EVCS bus locations (from config)
    """
    # Handle backward compatibility: if scalar is passed, distribute equally
    if isinstance(ev_loads_per_bus, (int, float)):
        # Single value - distribute equally across all buses (legacy behavior)
        per_bus_load = ev_loads_per_bus / len(evcs_bus) if evcs_bus else 0
        ev_loads_per_bus = {bus: per_bus_load for bus in evcs_bus}

    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    master_dss_path = os.path.join(script_dir, 'master.dss')

    try:
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

        # Add EV load at each EVCS bus (multi-bus support)
        for bus in evcs_bus:
            bus_name = bus.split('.')[0]
            ev_load_name = f"EV_{bus_name}"
            # Get load for THIS specific bus (multi-bus support)
            bus_load_kw = ev_loads_per_bus.get(bus, 0.0)

            # Check if EV load already exists
            if ev_load_name.lower() not in [n.lower() for n in dss.Loads.AllNames()]:
                # Create new EV load at this bus
                dss.Text.Command(f"New Load.{ev_load_name} Bus1={bus} kW={bus_load_kw} kvar=0 model=1")
            else:
                # Update existing EV load
                dss.Loads.Name(ev_load_name)
                dss.Loads.kW(bus_load_kw)

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

    except Exception as e:
        # If OpenDSS fails, return None to skip voltage checking
        return None


def uncontrolled_charging(initial_soc, num_control_steps, control_interval, battery_capacity,
                          charging_efficiency, arrival_time_idx, departure_time_idx,
                          num_evs, max_charging_rate, desired_state_of_charge):
    """
    Simulates uncontrolled charging: Charge at max rate until target SOC is reached.

    Args:
        initial_soc: Initial SOC for each EV
        num_control_steps: Number of time steps
        control_interval: Time step duration (hours)
        battery_capacity: Battery capacity (kWh)
        charging_efficiency: Charging efficiency (0-1)
        arrival_time_idx: Arrival time index for each EV
        departure_time_idx: Departure time index for each EV
        num_evs: Number of EVs
        max_charging_rate: Maximum charging rate (kW)
        desired_state_of_charge: Target SOC

    Returns:
        tuple: (soc, charging_rate) arrays
    """
    soc = np.zeros((num_evs, num_control_steps))
    charging_rate = np.zeros((num_evs, num_control_steps))
    soc_tolerance = 0.001

    # Set initial SOC at the arrival time step for each EV
    for ev in range(num_evs):
        if arrival_time_idx[ev] < num_control_steps:
            soc[ev, arrival_time_idx[ev]] = initial_soc[ev]
            if arrival_time_idx[ev] > 0:
                soc[ev, 0:arrival_time_idx[ev]] = initial_soc[ev]

    # Simulate step-by-step
    for t in range(num_control_steps - 1):
        for ev in range(num_evs):
            is_present = (t >= arrival_time_idx[ev] and t < departure_time_idx[ev])
            needs_charge = (soc[ev, t] < desired_state_of_charge - soc_tolerance)

            if is_present and needs_charge:
                charging_rate[ev, t] = max_charging_rate
            else:
                charging_rate[ev, t] = 0

            if is_present:
                charged_energy = charging_rate[ev, t] * control_interval * charging_efficiency
                soc[ev, t + 1] = soc[ev, t] + (charged_energy / battery_capacity)
            else:
                soc[ev, t + 1] = soc[ev, t]

            soc[ev, t + 1] = np.clip(soc[ev, t + 1], 0, desired_state_of_charge)

    charging_rate[:, -1] = 0
    return soc, charging_rate


def calculate_soc(initial_soc, charging_rate, num_control_steps, control_interval,
                  battery_capacity, charging_efficiency, arrival_time_idx, departure_time_idx, num_evs):
    """
    Calculate the State of Charge (SOC) for each EV over time steps.
    """
    soc = np.zeros((num_evs, num_control_steps))

    for ev in range(num_evs):
        if arrival_time_idx[ev] < num_control_steps:
            soc[ev, arrival_time_idx[ev]] = initial_soc[ev]
            if arrival_time_idx[ev] > 0:
                soc[ev, 0:arrival_time_idx[ev]] = initial_soc[ev]

    for t in range(num_control_steps - 1):
        for ev in range(num_evs):
            if t >= arrival_time_idx[ev] and t < departure_time_idx[ev]:
                charged_energy = charging_rate[ev, t] * control_interval * charging_efficiency
                soc[ev, t + 1] = soc[ev, t] + (charged_energy / battery_capacity)
            else:
                soc[ev, t + 1] = soc[ev, t]
            soc[ev, t + 1] = np.clip(soc[ev, t + 1], 0, 1)

    return soc


def simulate_real_charging_process(initial_soc, scheduled_charging_rate, num_control_steps,
                                    control_interval, battery_capacity, charging_efficiency,
                                    arrival_time_idx, departure_time_idx, num_evs):
    """
    Simulates REAL battery physics. If scheduled rate would overcharge,
    actual rate is reduced. Returns (soc, real_charging_rate).

    This function ensures:
    1. Battery cannot exceed 100% SOC
    2. Real charging rate is back-calculated for accurate cost
    3. Cost is based on actual energy charged, not scheduled
    """
    soc = np.zeros((num_evs, num_control_steps))
    real_charging_rate = np.zeros((num_evs, num_control_steps))

    # Initialize SOC
    for ev in range(num_evs):
        if arrival_time_idx[ev] < num_control_steps:
            soc[ev, :arrival_time_idx[ev]+1] = initial_soc[ev]

    for t in range(num_control_steps - 1):
        for ev in range(num_evs):
            if t >= arrival_time_idx[ev] and t < departure_time_idx[ev]:
                current_soc = soc[ev, t]
                max_energy_input = (1.0 - current_soc) * battery_capacity
                attempted_energy = scheduled_charging_rate[ev, t] * control_interval * charging_efficiency
                actual_energy = max(0, min(attempted_energy, max_energy_input))

                if actual_energy > 0:
                    real_charging_rate[ev, t] = actual_energy / (control_interval * charging_efficiency)
                else:
                    real_charging_rate[ev, t] = 0

                soc[ev, t + 1] = current_soc + (actual_energy / battery_capacity)
            else:
                real_charging_rate[ev, t] = 0
                soc[ev, t + 1] = soc[ev, t]

            soc[ev, t + 1] = np.clip(soc[ev, t + 1], 0, 1.0)

    return soc, real_charging_rate


def calculate_cost(charging_rate, electricity_price, num_control_steps, control_interval,
                   num_evs, arrival_time_idx, departure_time_idx):
    """Calculate the total charging cost."""
    total_cost = 0
    for t in range(num_control_steps):
        for ev in range(num_evs):
            if t >= arrival_time_idx[ev] and t < departure_time_idx[ev]:
                total_cost += charging_rate[ev, t] * control_interval * electricity_price[t]
    return total_cost


def calculate_cost_per_step(charging_rate, electricity_price, num_control_steps, control_interval,
                            num_evs, arrival_time_idx, departure_time_idx):
    """Calculate the charging cost incurred at each time step across all EVs."""
    cost_per_step = np.zeros(num_control_steps)
    for t in range(num_control_steps):
        step_cost = 0
        for ev in range(num_evs):
            if t >= arrival_time_idx[ev] and t < departure_time_idx[ev]:
                step_cost += charging_rate[ev, t] * control_interval * electricity_price[t]
        cost_per_step[t] = step_cost
    return cost_per_step


def fitness_function(charging_rate, feeder_loads_p, feeder_loads_q, load_ids, evcs_bus, ev_params=None):
    """
    Fitness function for PSO. Minimize cost while meeting SOC and grid voltage requirements.

    Improvements:
    - Uses physics-based battery simulation for accurate cost
    - Proportional penalty (2000 x missing SOC) instead of binary
    - Cost based on REAL energy charged, not scheduled

    Args:
        charging_rate: Charging schedule array (num_evs, num_control_steps)
        feeder_loads_p: Dict of bus_id -> real power from feeder
        feeder_loads_q: Dict of bus_id -> reactive power from feeder
        load_ids: List of load bus IDs (from PowersReal.ids)
        evcs_bus: List of EVCS bus locations (from config)
        ev_params: Optional dict of EV parameters. If None, imports from ev_parameters.
    """
    # Get parameters from ev_params dict or import from module (backward compatible)
    if ev_params is not None:
        num_evs = ev_params["num_evs"]
        num_control_steps = ev_params["num_control_steps"]
        control_interval = ev_params["control_interval"]
        battery_capacity = ev_params["battery_capacity"]
        charging_efficiency = ev_params["charging_efficiency"]
        electricity_price = ev_params["electricity_price"]
        arrival_time_idx = ev_params["arrival_time_idx"]
        departure_time_idx = ev_params["departure_time_idx"]
        initial_soc = ev_params["initial_soc"]
        desired_state_of_charge = ev_params["desired_soc"]
        evcs_bus_assignment = ev_params["evcs_bus_assignment"]
    else:
        # Backward compatibility: import from ev_parameters module
        from ev_parameters import (
            num_evs, num_control_steps, control_interval, battery_capacity,
            charging_efficiency, electricity_price, arrival_time_idx,
            departure_time_idx, initial_soc, desired_state_of_charge,
            evcs_bus_assignment
        )

    # Use physics engine for accurate cost calculation
    real_soc, real_rate = simulate_real_charging_process(
        initial_soc, charging_rate, num_control_steps, control_interval,
        battery_capacity, charging_efficiency, arrival_time_idx, departure_time_idx, num_evs
    )

    # Calculate cost based on REAL rate (not scheduled)
    cost = calculate_cost(real_rate, electricity_price, num_control_steps, control_interval,
                          num_evs, arrival_time_idx, departure_time_idx)

    # --- Penalties ---
    undershoot_penalty = 0
    voltage_penalty = 0

    # Proportional penalty (provides gradient information)
    penalty_weight = 2000
    for ev in range(num_evs):
        dep_check_idx = min(departure_time_idx[ev] - 1, num_control_steps - 1)
        if dep_check_idx >= 0:
            final_soc = real_soc[ev, dep_check_idx]
            if final_soc < desired_state_of_charge - 0.01:
                undershoot_penalty += penalty_weight * (desired_state_of_charge - final_soc)

    # Penalty for voltage violations (multi-bus support)
    for t in range(num_control_steps):
        ev_loads_per_bus = {}
        for bus, ev_indices in evcs_bus_assignment.items():
            if ev_indices:
                bus_load = np.sum(real_rate[ev_indices, t])
            else:
                bus_load = 0.0
            ev_loads_per_bus[bus] = bus_load

        voltages = run_opendss_simulation(ev_loads_per_bus, feeder_loads_p, feeder_loads_q, load_ids, evcs_bus)
        if voltages is not None:
            if np.any(voltages > 1.05) or np.any(voltages < 0.95):
                voltage_penalty += 1000

    return cost + undershoot_penalty + voltage_penalty


def ev_pso_optimization(num_particles, max_iterations, feeder_loads_p, feeder_loads_q,
                        load_ids, evcs_bus, ev_params=None):
    """
    Performs Particle Swarm Optimization to find the optimal charging schedule for electric vehicles.

    Improvements:
    - Hot-start: Seeds first particle with baseline (uncontrolled) solution
    - Inertia decay: 0.8 decaying by 0.99 each iteration
    - Velocity clipping: +/-20% instead of +/-10%
    - Returns true cost (without penalties)

    Args:
        num_particles: Number of particles in the swarm
        max_iterations: Maximum number of iterations
        feeder_loads_p: Dict of bus_id -> real power from feeder
        feeder_loads_q: Dict of bus_id -> reactive power from feeder
        load_ids: List of load bus IDs (from PowersReal.ids)
        evcs_bus: List of EVCS bus locations (from config)
        ev_params: Optional dict of EV parameters. If None, imports from ev_parameters.

    Returns:
        tuple: (global_best_position, true_cost)
    """
    # Get parameters from ev_params dict or import from module (backward compatible)
    if ev_params is not None:
        num_evs = ev_params["num_evs"]
        max_charging_rate = ev_params["max_charging_rate"]
        charging_efficiency = ev_params["charging_efficiency"]
        battery_capacity = ev_params["battery_capacity"]
        control_interval = ev_params["control_interval"]
        num_control_steps = ev_params["num_control_steps"]
        electricity_price = ev_params["electricity_price"]
        arrival_time_idx = ev_params["arrival_time_idx"]
        departure_time_idx = ev_params["departure_time_idx"]
        initial_soc = ev_params["initial_soc"]
        desired_state_of_charge = ev_params["desired_soc"]
        charging_energy = ev_params["charging_energy"]
    else:
        # Backward compatibility: import from ev_parameters module
        from ev_parameters import (
            num_evs, max_charging_rate, charging_efficiency, battery_capacity,
            control_interval, num_control_steps, electricity_price,
            arrival_time_idx, departure_time_idx, initial_soc,
            desired_state_of_charge, charging_energy
        )

    print("  [PSO] ========== PSO INITIALIZATION ==========")
    print(f"  [PSO] EVs: {num_evs}, Time steps: {num_control_steps}, Max rate: {max_charging_rate} kW")
    print(f"  [PSO] Target SOC: {desired_state_of_charge*100:.0f}%")
    print(f"  [PSO] Initial SOC range: {np.min(initial_soc)*100:.1f}% - {np.max(initial_soc)*100:.1f}%")
    print(f"  [PSO] Particles: {num_particles}, Iterations: {max_iterations}")
    print("  [PSO] Using Hot Start (Baseline Strategy)...")

    # Hot-start - get baseline (uncontrolled) solution first
    baseline_soc, baseline_rate = uncontrolled_charging(
        initial_soc, num_control_steps, control_interval, battery_capacity,
        charging_efficiency, arrival_time_idx, departure_time_idx,
        num_evs, max_charging_rate, desired_state_of_charge
    )

    baseline_total_energy = np.sum(baseline_rate) * control_interval
    print(f"  [PSO] Baseline total energy: {baseline_total_energy:.1f} kWh")

    # Initialize particles randomly
    particles = np.random.uniform(0, max_charging_rate, (num_particles, num_evs, num_control_steps))

    # Inject baseline as first particle
    particles[0] = baseline_rate.copy()

    # Seed nearby particles with noisy baseline
    num_seeded = min(10, num_particles - 1)
    for p in range(1, num_seeded + 1):
        noise = np.random.normal(0, max_charging_rate * 0.1, baseline_rate.shape)
        particles[p] = np.clip(baseline_rate + noise, 0, max_charging_rate)

    print(f"  [PSO] Seeded {num_seeded + 1} particles with baseline strategy")

    # Ensure all particles respect arrival/departure times
    for p in range(num_particles):
        for ev in range(num_evs):
            particles[p, ev, :arrival_time_idx[ev]] = 0
            particles[p, ev, departure_time_idx[ev]:] = 0

    # Velocity clipping +/-20%
    velocities = np.random.uniform(-max_charging_rate * 0.2, max_charging_rate * 0.2,
                                    (num_particles, num_evs, num_control_steps))

    print("  [PSO] Evaluating initial particle fitnesses...")

    # Initial evaluation
    personal_best_positions = particles.copy()
    personal_best_fitnesses = np.array([
        fitness_function(particles[i], feeder_loads_p, feeder_loads_q, load_ids, evcs_bus, ev_params)
        for i in range(num_particles)
    ])
    global_best_index = np.argmin(personal_best_fitnesses)
    global_best_position = personal_best_positions[global_best_index].copy()
    global_best_fitness = personal_best_fitnesses[global_best_index]

    print(f"  [PSO] Baseline fitness (Particle 0): {personal_best_fitnesses[0]:.2f}")
    print(f"  [PSO] Initial best fitness: {global_best_fitness:.2f} (Particle {global_best_index})")
    print("  [PSO] ========== OPTIMIZATION STARTING ==========")

    # PSO parameters with inertia decay
    inertia_weight = 0.8
    cognitive_coefficient = 1.4
    social_coefficient = 1.4
    decay = 0.99

    for iteration in range(max_iterations):
        for i in range(num_particles):
            r1 = np.random.rand(num_evs, num_control_steps)
            r2 = np.random.rand(num_evs, num_control_steps)
            velocities[i] = (inertia_weight * velocities[i] +
                             cognitive_coefficient * r1 * (personal_best_positions[i] - particles[i]) +
                             social_coefficient * r2 * (global_best_position - particles[i]))

            velocities[i] = np.clip(velocities[i], -max_charging_rate * 0.2, max_charging_rate * 0.2)
            particles[i] = particles[i] + velocities[i]

            # Apply constraints
            for ev in range(num_evs):
                particles[i, ev, :arrival_time_idx[ev]] = 0
                particles[i, ev, departure_time_idx[ev]:] = 0
                particles[i, ev, arrival_time_idx[ev]:departure_time_idx[ev]] = np.clip(
                    particles[i, ev, arrival_time_idx[ev]:departure_time_idx[ev]], 0, max_charging_rate
                )

                # Prevent overcharging
                cumulated_energy = 0
                for j in range(num_control_steps):
                    cumulated_energy = cumulated_energy + particles[i, ev, j] * control_interval
                    if cumulated_energy > 1.05 * charging_energy[ev]:
                        particles[i, ev, j+1:] = 0
                        break

            # Evaluate fitness
            current_fitness = fitness_function(particles[i], feeder_loads_p, feeder_loads_q,
                                               load_ids, evcs_bus, ev_params)

            if current_fitness < personal_best_fitnesses[i]:
                personal_best_positions[i] = particles[i].copy()
                personal_best_fitnesses[i] = current_fitness

                if current_fitness < global_best_fitness:
                    global_best_position = particles[i].copy()
                    global_best_fitness = current_fitness

        inertia_weight *= decay

        if (iteration + 1) % 5 == 0 or iteration == max_iterations - 1:
            print(f"  [PSO] Iter {iteration + 1:2d}/{max_iterations}: Best={global_best_fitness:.2f}, Inertia={inertia_weight:.3f}")

    print("  [PSO] ========== OPTIMIZATION COMPLETE ==========")
    print(f"  [PSO] Final optimized fitness: {global_best_fitness:.2f}")

    # Return TRUE cost (without penalties)
    final_soc, real_final_rates = simulate_real_charging_process(
        initial_soc, global_best_position, num_control_steps, control_interval,
        battery_capacity, charging_efficiency, arrival_time_idx, departure_time_idx, num_evs
    )

    evs_at_target = 0
    for ev in range(num_evs):
        dep_idx = min(departure_time_idx[ev] - 1, num_control_steps - 1)
        if final_soc[ev, dep_idx] >= desired_state_of_charge - 0.01:
            evs_at_target += 1
    print(f"  [PSO] EVs reaching target SOC: {evs_at_target}/{num_evs}")

    final_true_cost = calculate_cost(
        real_final_rates, electricity_price, num_control_steps,
        control_interval, num_evs, arrival_time_idx, departure_time_idx
    )

    optimized_energy = np.sum(real_final_rates) * control_interval
    print(f"  [PSO] Total energy scheduled: {optimized_energy:.1f} kWh")
    print(f"  [PSO] True electricity cost: ${final_true_cost:.2f}")
    print("  [PSO] ============================================")

    return global_best_position, final_true_cost
