# OEDI SI: EV Charging Scheduling Use Case (ORNL)

## Overview

The `evcs-ieee123` use case simulates an EV charging station connected to an IEEE 123-bus distribution network. The core of the simulation is the `evcs_federate`, a co-simulation component that encapsulates a Particle Swarm Optimization (PSO) algorithm.

## How to Run the Simulation

### Local Testing 

```bash
# Install dependencies
pip install -r LocalFeeder/requirements.txt

# Build the simulation
oedisi build --system system.json --component-dict components.json

# Run the simulation
oedisi run
```

### Docker 

```bash
# Build Docker image
docker build -t oedisi-evcs-v1:latest .

# Run simulation in Docker
docker run --rm oedisi-evcs-v1:latest bash -c \
  "cd /simulation && oedisi build --system system.json --component-dict components.json && oedisi run"
```

## Checking Results

After running the simulation:

```bash
# Check EVCS federate log for PSO results
cat build/evcs.log

# Check feeder log for power flow results
cat build/feeder.log

# View recorded data (if simulation completed)
ls build/recorder_*/
```

## Configuration

### Simulation Parameters (`system.json`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `number_of_timesteps` | 96 | Simulation duration (15-min intervals = 24 hours) |
| `run_freq_sec` | 900 | Time step in seconds |

### PSO Parameters (`evcs_federate/evcs_federate.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_particles` | 150 | Number of PSO particles |
| `max_iterations` | 30 | PSO iterations per timestep |


