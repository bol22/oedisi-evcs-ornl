# OEDI SI: EV Charging Scheduling Use Case (ORNL)

This repository contains the OEDI SI use case for grid-aware electric vehicle (EV) charging scheduling, developed by Oak Ridge National Laboratory (ORNL). It demonstrates how a flexible load management algorithm can be integrated as a co-simulation component to interact with a power distribution system model.

## Use Case Overview

The `evcs-ieee123` use case simulates an EV charging station connected to an IEEE 123-bus distribution network. The core of the simulation is the `evcs_federate`, a co-simulation component that encapsulates a Particle Swarm Optimization (PSO) algorithm. At each time step of the simulation, this federate determines the optimal charging rate for the entire EV station with two main objectives:

1.  **Minimize Electricity Costs:** The optimization considers time-varying electricity prices to charge the vehicles when energy is cheapest.
2.  **Ensure Grid Stability:** The federate includes an internal model of the IEEE 123-bus network. Before deciding on a final charging rate, it runs its own internal power flow simulations to ensure that the proposed EV load will not cause any voltage violations on the grid.

This "internal model" architecture allows the `evcs_federate` to operate as an autonomous, grid-aware agent, sending only a final, validated control action to the main power grid simulation.

## Co-simulation Architecture

This use case is built on the OEDISI single-container framework, which uses HELICS for co-simulation. The simulation consists of three main components (federates):

*   **`LocalFeeder`:** Simulates the IEEE 123-bus distribution network using OpenDSS. At each time step, it publishes the current state of the grid (voltages, power flows) and subscribes to the EV load from the `evcs_federate`.
*   **`evcs_federate`:** Subscribes to the grid state from the `LocalFeeder`. It uses this information as the baseline for its internal, grid-aware PSO algorithm to determine the optimal and safest charging load for the EV station. It then publishes this load back to the `LocalFeeder`.
*   **`recorder`:**  Subscribes to various data streams from the other federates to log the results of the simulation for later analysis.

## How to Run the Simulation

This use case is designed to be run as a single-container simulation using the `oedisi` command-line interface.

### Prerequisites

*   Docker
*   Python 3.9+
*   The `oedisi` Python package (`pip install oedisi`)

### Running the Simulation

1.  **Build the Simulation Environment:**
    From the root of this directory (`oedisi-example-evcs-ieee123`), run the `oedisi build` command. This will parse the `system.json` and `components.json` files and create a `build` directory with the complete, runnable simulation configuration.

    ```bash
    oedisi build --system system.json --component-dict components.json
    ```

2.  **Run**
    Run the `oedisi run` command. This will automatically start the HELICS broker and all the federates in the correct order to execute the co-simulation.

    ```bash
    oedisi run
    ```

