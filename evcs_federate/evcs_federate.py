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
import ev_parameters
import numpy as np

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


class EVCSFederate:
    "EVCS federate. Wraps EV simulation with pubs and subs"

    def __init__(
        self,
        federate_name,
        input_mapping,
        broker_config: BrokerConfig,
    ):
        "Initializes federate with name and remaps input into subscriptions"
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
        self.vfed.enter_executing_mode()
        granted_time = h.helicsFederateRequestTime(self.vfed, h.HELICS_TIME_MAXTIME)
        
        num_particles = 150
        max_iterations = 30

        while granted_time < h.HELICS_TIME_MAXTIME:
            if not self.sub_power_P.is_updated():
                granted_time = h.helicsFederateRequestTime(
                    self.vfed, h.HELICS_TIME_MAXTIME
                )
                continue

            charging_rate, _ = ev_simulation.ev_pso_optimization(num_particles, max_iterations)
            total_charging_power = np.sum(charging_rate, axis = 0)
            
            time = self.sub_power_P.json.time

            ev_load_real = PowersReal(
                ids=ev_parameters.evcs_bus,
                values=[total_charging_power[granted_time]],
                time=time
            )
            ev_load_imag = PowersImaginary(
                ids=ev_parameters.evcs_bus,
                values=[0.0],
                time=time
            )

            self.pub_ev_load_real.publish(ev_load_real.json())
            self.pub_ev_load_imag.publish(ev_load_imag.json())
            logger.info(f"Published EV load: {ev_load_real.values[0]} kW")

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

    with open("input_mapping.json") as f:
        input_mapping = json.load(f)

    sfed = EVCSFederate(
        federate_name, input_mapping, broker_config
    )

    try:
        sfed = EVCSFederate(
        federate_name, input_mapping, broker_config
        )
        logger.info("Value federate created")
    except h.HelicsException as e:
        logger.error(f"Failed to create HELICS Value Federate: {str(e)}")
        
    sfed.run()
    logger.info(f"Running------------------------------------------------")

if __name__ == "__main__":
    run_simulator(BrokerConfig(broker_ip="127.0.0.1"))