import logging
import helics as h
import json
from datetime import datetime
from oedisi.types.common import BrokerConfig
from oedisi.types.data_types import (
    PowersImaginary,
    PowersReal,
)
import ev_simulation
from ev_parameters import generate_ev_parameters
import numpy as np

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


# Known 3-phase loads in gadal_ieee123 (bus has S{num} not S{num}a/b/c)
THREE_PHASE_BUSES = {'47', '48'}


def bus_id_to_load_name(bus_id):
    """Convert bus ID (e.g., '48.1') to OpenDSS load name.

    Handles two cases based on gadal_ieee123 model:
    - Single-phase buses: '65.1' -> 'S65a' (phase 1->a, 2->b, 3->c)
    - Three-phase buses: '48.1' -> 'S48' (no phase suffix)
    """
    phase_map = {'1': 'a', '2': 'b', '3': 'c'}
    try:
        parts = bus_id.split('.')
        if len(parts) == 2:
            bus_num = parts[0]
            phase_num = parts[1]

            # Check if this is a 3-phase bus (no phase suffix needed)
            if bus_num in THREE_PHASE_BUSES:
                return f"S{bus_num}"

            # Single-phase bus: add phase letter
            phase_letter = phase_map.get(phase_num, 'a')
            return f"S{bus_num}{phase_letter}"
    except Exception:
        pass
    return bus_id  # Return original if conversion fails


class EVCSFederate:
    "EVCS federate. Wraps EV simulation with pubs and subs"

    def __init__(
        self,
        federate_name,
        input_mapping,
        broker_config: BrokerConfig,
        evcs_bus: list = None,
        ev_params: dict = None,
    ):
        "Initializes federate with name and remaps input into subscriptions"
        # Store evcs_bus configuration (default to ["48.1"] if not provided)
        self.evcs_bus = evcs_bus if evcs_bus is not None else ["48.1"]
        logger.info(f"EVCS bus location(s): {self.evcs_bus}")

        # Store EV parameters (generated from config or defaults)
        self.ev_params = ev_params
        if ev_params is not None:
            logger.info(f"Loaded EV config: {ev_params['num_evs']} EVs, "
                       f"{ev_params['num_control_steps']} control steps")

        deltat = 1

        fedinfo = h.helicsCreateFederateInfo()

        h.helicsFederateInfoSetBroker(fedinfo, broker_config.broker_ip)
        h.helicsFederateInfoSetBrokerPort(fedinfo, broker_config.broker_port)

        fedinfo.core_name = federate_name
        fedinfo.core_type = h.HELICS_CORE_TYPE_ZMQ
        fedinfo.core_init = "--federates=1"
        h.helicsFederateInfoSetTimeProperty(
            fedinfo, h.helics_property_time_delta, deltat
        )

        self.vfed = h.helicsCreateValueFederate(federate_name, fedinfo)
        logger.info("Value federate created")

        # Register the publication #
        self.sub_power_P = self.vfed.register_subscription(
            input_mapping["powers_real_in"], "W"
        )
        self.sub_power_Q = self.vfed.register_subscription(
            input_mapping["powers_imag_in"], "W"
        )
        self.pub_ev_load_real = self.vfed.register_publication(
            "ev_load_real", h.HELICS_DATA_TYPE_STRING, ""
        )
        self.pub_ev_load_imag = self.vfed.register_publication(
            "ev_load_imag", h.HELICS_DATA_TYPE_STRING, ""
        )

    def run(self):
        logger.info(f"Federate connected: {datetime.now()}")
        logger.info("=" * 60)
        logger.info("EVCS FEDERATE STARTING - Multi-Bus Mode")
        logger.info(f"  Target buses: {self.evcs_bus}")

        # Get bus assignment from ev_params or default
        evcs_bus_assignment = (self.ev_params.get("evcs_bus_assignment", {})
                               if self.ev_params else {})
        for bus, evs in evcs_bus_assignment.items():
            if evs:
                logger.info(f"    Bus {bus}: {len(evs)} EVs (indices {evs[0]}-{evs[-1]})")
        logger.info("=" * 60)

        self.vfed.enter_executing_mode()
        granted_time = h.helicsFederateRequestTime(self.vfed, h.HELICS_TIME_MAXTIME)

        # Reduced for faster Docker simulation (original: 150 particles, 30 iterations)
        num_particles = 30
        max_iterations = 10
        timestep_count = 0

        while granted_time < h.HELICS_TIME_MAXTIME:
            if not self.sub_power_P.is_updated():
                granted_time = h.helicsFederateRequestTime(
                    self.vfed, h.HELICS_TIME_MAXTIME
                )
                continue

            timestep_count += 1
            logger.info("")
            logger.info("=" * 60)
            logger.info(f"TIMESTEP {timestep_count} | HELICS Time: {granted_time}")
            logger.info("=" * 60)

            power_P = PowersReal.parse_obj(self.sub_power_P.json)
            power_Q = PowersImaginary.parse_obj(self.sub_power_Q.json)

            # Extract load_ids dynamically from PowersReal (instead of hardcoding)
            load_ids = list(power_P.ids)
            logger.info(f"[INPUT] Received feeder data: {len(load_ids)} load buses")

            # Show sample of feeder loads
            sample_loads = list(zip(power_P.ids[:3], power_P.values[:3]))
            logger.info(f"[INPUT] Sample loads (first 3): {sample_loads}")

            feeder_loads_p = {bus: p for bus, p in zip(power_P.ids, power_P.values)}
            feeder_loads_q = {bus: q for bus, q in zip(power_Q.ids, power_Q.values)}

            # Pass load_ids and evcs_bus to PSO optimization
            logger.info(f"[PSO] Starting optimization: {num_particles} particles, {max_iterations} iterations")
            import time as time_module
            pso_start = time_module.time()

            charging_rate, true_cost = ev_simulation.ev_pso_optimization(
                num_particles, max_iterations, feeder_loads_p, feeder_loads_q,
                load_ids, self.evcs_bus, ev_params=self.ev_params
            )

            pso_duration = time_module.time() - pso_start
            logger.info(f"[PSO] Optimization complete in {pso_duration:.2f} seconds")
            logger.info(f"[PSO] True electricity cost: ${true_cost:.2f}")

            time = power_P.time

            # Convert granted_time to integer index for array access
            time_idx = int(granted_time)

            # Calculate per-bus charging power (multi-bus support)
            ev_load_values = []
            ev_load_per_bus = {}
            for bus in self.evcs_bus:
                # Get EV indices assigned to this bus
                ev_indices = evcs_bus_assignment.get(bus, [])
                if ev_indices:
                    # Sum charging rates for EVs at this bus at current timestep
                    bus_power = float(np.sum(charging_rate[ev_indices, time_idx]))
                    num_charging = int(np.sum(charging_rate[ev_indices, time_idx] > 0))
                else:
                    bus_power = 0.0
                    num_charging = 0
                ev_load_values.append(bus_power)
                ev_load_per_bus[bus] = bus_power
                logger.info(f"[RESULT] Bus {bus}: {num_charging} EVs charging, Power: {bus_power:.2f} kW")

            total_ev_load = sum(ev_load_values)
            logger.info(f"[RESULT] Total across all buses: {total_ev_load:.2f} kW")

            # Convert bus IDs to OpenDSS load names (e.g., "48.1" -> "S48a")
            load_names = [bus_id_to_load_name(bus) for bus in self.evcs_bus]
            equipment_ids = [bus.split('.')[0] for bus in self.evcs_bus]

            # Publish per-bus EV loads using OpenDSS load names
            ev_load_real = PowersReal(
                ids=load_names,  # Use load names, not bus IDs
                equipment_ids=equipment_ids,
                values=ev_load_values,  # Per-bus values, not single total
                time=time
            )
            ev_load_imag = PowersImaginary(
                ids=load_names,  # Use load names, not bus IDs
                equipment_ids=equipment_ids,
                values=[0.0] * len(self.evcs_bus),  # Zero reactive power per bus
                time=time
            )
            logger.info(f"[OUTPUT] Load name mapping: {dict(zip(self.evcs_bus, load_names))}")

            self.pub_ev_load_real.publish(ev_load_real.json())
            self.pub_ev_load_imag.publish(ev_load_imag.json())
            logger.info(f"[OUTPUT] Published to HELICS: {ev_load_per_bus}")
            logger.info("-" * 60)

        self.stop()

    def stop(self):
        h.helicsFederateDisconnect(self.vfed)
        h.helicsFederateFree(self.vfed)
        h.helicsCloseLibrary()
        logger.info(f"Federate disconnected: {datetime.now()}")


def run_simulator(broker_config: BrokerConfig):
    logger.info(f"Running---------------------------------------------------")
    with open("static_inputs.json") as f:
        config = json.load(f)
        federate_name = config["name"]
        # Read evcs_bus from config (with default fallback)
        evcs_bus = config.get("evcs_bus", ["48.1"])
        logger.info(f"Loaded evcs_bus from config: {evcs_bus}")

    # Generate EV parameters from configuration
    # This uses config values or falls back to DEFAULT_CONFIG
    ev_params = generate_ev_parameters(config)
    logger.info(f"Generated EV parameters: {ev_params['num_evs']} EVs across "
                f"{len(ev_params['num_evs_per_station'])} stations")

    with open("input_mapping.json") as f:
        input_mapping = json.load(f)

    try:
        sfed = EVCSFederate(
            federate_name, input_mapping, broker_config, evcs_bus,
            ev_params=ev_params
        )
        logger.info("Value federate created")
    except h.HelicsException as e:
        logger.error(f"Failed to create HELICS Value Federate: {str(e)}")
        return

    sfed.run()
    logger.info(f"Running------------------------------------------------")

if __name__ == "__main__":
    run_simulator(BrokerConfig(broker_ip="127.0.0.1"))