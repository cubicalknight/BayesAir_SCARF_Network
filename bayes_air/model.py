"""Define a probabilistic model for an air traffic network."""
from copy import deepcopy

import pyro
import pyro.distributions as dist
import torch

from bayes_air.network import NetworkState, AugmentedNetworkState
from bayes_air.types import QueueEntry, DepartureQueueEntry, AirportCode

FAR_FUTURE_TIME = 30.0


def air_traffic_network_model(
    states: list[NetworkState],
    delta_t: float = 0.1,
    max_t: float = FAR_FUTURE_TIME,
    device=None,
    include_cancellations: bool = False,
):
    """
    Simulate the behavior of an air traffic network.

    Args:
        states: the starting states of the simulation (will run an independent
            simulation from each start state). All states must include the same
            airports.
        delta_t: the time resolution of the simulation, in hours
        max_t: the maximum time to simulate, in hours
        device: the device to run the simulation on
        include_cancellations: whether to include the possibility of flight
            cancellations (if False, crew/aircraft reserves will not be modeled)
    """
    if device is None:
        device = torch.device("cpu")

    # Copy state to avoid modifying it
    states = deepcopy(states)

    # Define system-level parameters
    runway_use_time_std_dev = pyro.param(
        "runway_use_time_std_dev",
        torch.tensor(0.1, device=device),  # used to be 0.025
        constraint=dist.constraints.positive,
    )
    travel_time_variation = pyro.param(
        "travel_time_variation",
        torch.tensor(0.1, device=device),  # used to be 0.05
        constraint=dist.constraints.positive,
    )
    turnaround_time_variation = pyro.param(
        "turnaround_time_variation",
        torch.tensor(0.1, device=device),  # used to be 0.05
        constraint=dist.constraints.positive,
    )

    # Sample latent variables for airports.
    airport_codes = states[0].airports.keys()
    airport_turnaround_times = {
        code: pyro.sample(
            f"{code}_mean_turnaround_time",
            dist.Gamma(
                torch.tensor(1.0, device=device), torch.tensor(2.0, device=device)
            ),
        )
        for code in airport_codes
    }
    airport_service_times = {
        code: pyro.sample(
            f"{code}_mean_service_time",
            dist.Gamma(
                torch.tensor(1.5, device=device), torch.tensor(10.0, device=device)
            ),
        )
        for code in airport_codes
    }
    travel_times = {
        (origin, destination): pyro.sample(
            f"travel_time_{origin}_{destination}",
            dist.Gamma(
                torch.tensor(4.0, device=device), torch.tensor(1.25, device=device)
            ),
        )
        for origin in airport_codes
        for destination in airport_codes
        if origin != destination
    }

    if include_cancellations:
        airport_initial_available_aircraft = {
            code: torch.exp(
                pyro.sample(
                    f"{code}_log_initial_available_aircraft",
                    dist.Normal(
                        torch.tensor(0.0, device=device),
                        torch.tensor(1.0, device=device),
                    ),
                )
            )
            for code in airport_codes
        }
        airport_base_cancel_prob = {
            code: torch.exp(
                pyro.sample(
                    f"{code}_base_cancel_logprob",
                    dist.Normal(
                        torch.tensor(-3.0, device=device),
                        torch.tensor(1.0, device=device),
                    ),
                )
            )
            for code in airport_codes
        }
    else:
        # To ignore cancellations, just provide practcially infinite reserves
        airport_initial_available_aircraft = {
            code: torch.tensor(1000.0, device=device) for code in airport_codes
        }
        airport_base_cancel_prob = {
            code: torch.tensor(0.0, device=device) for code in airport_codes
        }

    # Simulate for each state
    output_states = []
    # for day_ind in pyro.plate("days", len(states)):
    for day_ind in pyro.markov(range(len(states)), history=1):
        state = states[day_ind]
        var_prefix = f"day{day_ind}_"

        # print(f"============= Starting day {day_ind} =============")
        # print(f"# pending flights: {len(state.pending_flights)}")
        # print(f"Initial aircraft: {airport_initial_available_aircraft}")
        # print(f"# in-transit flights: {len(state.in_transit_flights)}")
        # print(f"# completed flights: {len(state.completed_flights)}")
        # print("Travel times:")
        # print(travel_times)

        # Assign the latent variables to the airports
        for airport in state.airports.values():
            airport.mean_service_time = airport_service_times[airport.code]
            airport.runway_use_time_std_dev = runway_use_time_std_dev
            airport.mean_turnaround_time = airport_turnaround_times[airport.code]
            airport.turnaround_time_std_dev = (
                turnaround_time_variation * airport.mean_turnaround_time
            )
            airport.base_cancel_prob = airport_base_cancel_prob[airport.code]

            # Initialize the available aircraft list
            airport.num_available_aircraft = airport_initial_available_aircraft[
                airport.code
            ]
            i = 0
            while i < airport.num_available_aircraft:
                airport.available_aircraft.append(torch.tensor(0.0, device=device))
                i += 1

        # Simulate the movement of aircraft within the system for a fixed period of time
        t = torch.tensor(0.0, device=device)
        while not state.complete:
            # Update the current time
            t += delta_t

            # All parked aircraft that are ready to turnaround get serviced
            for airport in state.airports.values():
                airport.update_available_aircraft(t)

            # If the maximum time has elapsed, add lots of reserve aircraft at each
            # airport. This is artificial and only done to ensure that the simulation
            # terminates.
            if t >= max_t:
                # print(f"TIME'S UP! Adding reserve aircraft at time {t}")
                for airport in state.airports.values():
                    airport.num_available_aircraft = airport.num_available_aircraft + 1
                    airport.available_aircraft.append(t)

            # All flights that are able to depart get moved to the runway queue at their
            # origin airport
            ready_to_depart_flights, ready_times = state.pop_ready_to_depart_flights(
                t, var_prefix
            )
            for flight, ready_time in zip(ready_to_depart_flights, ready_times):
                queue_entry = QueueEntry(flight=flight, queue_start_time=ready_time)
                state.airports[flight.origin].runway_queue.append(queue_entry)

            # All flights that are using the runway get serviced
            for airport in state.airports.values():
                departed_flights, landing_flights = airport.update_runway_queue(
                    t, var_prefix
                )

                # Departing flights get added to the in-transit list, while landed flights
                # get added to the completed list
                state.add_in_transit_flights(
                    departed_flights, travel_times, travel_time_variation, var_prefix
                )
                state.add_completed_flights(landing_flights)

            # All flights that are in transit get moved to the runway queue at their
            # destination airport, if enough time has elapsed
            state.update_in_transit_flights(t)

        # print(f"---------- Completing day {day_ind} ----------")
        # print(f"# pending flights: {len(state.pending_flights)}")
        # print(f"# in-transit flights: {len(state.in_transit_flights)}")
        # print(f"# completed flights: {len(state.completed_flights)}")

        # Once we're done, return the state (this will include the actual arrival/departure
        # times for each aircraft)
        output_states.append(state)

    return output_states



def augmented_air_traffic_network_model(
    states: list[AugmentedNetworkState],
    empirical_travel_times: dict[tuple[AirportCode, AirportCode], float],
    delta_t: float = 0.1,
    max_t: float = FAR_FUTURE_TIME,
    device=None,
    include_cancellations: bool = True,
    obs_none: bool = False,
):
    """
    Simulate the behavior of an air traffic network.

    Args:
        states: the starting states of the simulation (will run an independent
            simulation from each start state). All states must include the same
            airports.x
        delta_t: the time resolution of the simulation, in hours
        max_t: the maximum time to simulate, in hours
        device: the device to run the simulation on
        include_cancellations: whether to include the possibility of flight
            cancellations (if False, crew/aircraft reserves will not be modeled)
    """
    if device is None:
        device = torch.device("cpu")

    # Copy state to avoid modifying it
    states = deepcopy(states)

    # set obs_none flags
    if obs_none:
        for state in states:
            state.obs_none = obs_none
            state.sync_child_obs_none()

    # Define system-level parameters
    runway_use_time_std_dev = pyro.param(
        "runway_use_time_std_dev",
        torch.tensor(0.1, device=device),  # used to be 0.025
        constraint=dist.constraints.positive,
    )
    travel_time_variation = pyro.param(
        "travel_time_variation",
        torch.tensor(0.1, device=device),  # used to be 0.05
        constraint=dist.constraints.positive,
    )
    turnaround_time_variation = pyro.param(
        "turnaround_time_variation",
        torch.tensor(0.1, device=device),  # used to be 0.05
        constraint=dist.constraints.positive,
    )

    # Sample latent variables for airports in network
    network_airport_codes = states[0].network_state.airports.keys()
    airport_turnaround_times = {
        # code: pyro.sample(
        #     f"{code}_mean_turnaround_time",
        #     dist.Gamma(
        #         torch.tensor(1.0, device=device), 
        #         torch.tensor(2.0*30, device=device)
        #     ),
        # )
        code: 0.1
        for code in network_airport_codes
    }
    shape = 1.0
    airport_service_times = {
        code: pyro.sample(
            f"{code}_mean_service_time",
            dist.Gamma(
                torch.tensor(shape, device=device), 
                torch.tensor(shape*80, device=device)
            ),
        )
        for code in network_airport_codes
    }
    network_travel_times = {
        (origin, destination): pyro.sample(
            f"travel_time_{origin}_{destination}",
            dist.Gamma(
                torch.tensor(4.0, device=device), 
                torch.tensor(1.25, device=device)
            ),
        )
        for origin in network_airport_codes
        for destination in network_airport_codes
        if origin != destination
    }

    if include_cancellations:
        network_airport_initial_available_aircraft = {
            code: torch.exp(
                pyro.sample(
                    f"{code}_log_initial_available_aircraft",
                    dist.Normal(
                        torch.tensor(0.0, device=device),
                        torch.tensor(1.0, device=device),
                    ),
                )
            )
            for code in network_airport_codes
        }
        network_airport_base_cancel_prob = {
            code: torch.exp(
                pyro.sample(
                    f"{code}_base_cancel_logprob",
                    dist.Normal(
                        torch.tensor(-3.0, device=device),
                        torch.tensor(1.0, device=device),
                    ),
                )
            )
            for code in network_airport_codes
        }
    else:
        # To ignore cancellations, just provide practcially infinite reserves
        airport_initial_available_aircraft = {
            code: torch.tensor(1000.0, device=device) 
            for code in network_airport_codes
        }
        airport_base_cancel_prob = {
            code: torch.tensor(0.0, device=device) 
            for code in network_airport_codes
        }

    # sample latent variables for variables outside network
    incoming_airport_codes = set()
    for state in states:
        current_codes = state.source_supernode.source_codes
        incoming_airport_codes.update(current_codes)
    incoming_airport_codes = list(incoming_airport_codes)

    # trying something
    shape = 4.0
    incoming_travel_times = {
        (origin, destination): 
            pyro.sample(
                f"travel_time_{origin}_{destination}",
                # dist.Gamma(
                #     torch.tensor(4.0, device=device), 
                #     torch.tensor(1.25, device=device)
                # )
                dist.Gamma(
                    torch.tensor(shape, device=device),
                    torch.tensor(
                        shape / (.9 * empirical_travel_times[(origin, destination)]), 
                        device=device
                    )
                )
            )
            # .9 * empirical_travel_times[(origin, destination)]
        for origin in incoming_airport_codes
        for destination in network_airport_codes
    }

    travel_times = network_travel_times | incoming_travel_times
    airport_initial_available_aircraft = network_airport_initial_available_aircraft
    airport_base_cancel_prob = network_airport_base_cancel_prob

    # Simulate for each state
    output_states = []
    # for day_ind in pyro.plate("days", len(states)):
    for day_ind in pyro.markov(range(len(states)), history=1):
        state = states[day_ind]
        # var_prefix = f"day{day_ind}_"
        var_prefix = f"{state.day_str}_"

        # print(f"============= Starting day {day_ind} =============")
        # print(f"# pending flights: {len(state.pending_flights)}")
        # print(f"Initial aircraft: {airport_initial_available_aircraft}")
        # print(f"# in-transit flights: {len(state.in_transit_flights)}")
        # print(f"# completed flights: {len(state.completed_flights)}")
        # print("Travel times:")
        # print(travel_times)

        # Assign the latent variables to the airports
        for airport in state.network_state.airports.values():
            airport.mean_service_time = airport_service_times[airport.code]
            airport.runway_use_time_std_dev = runway_use_time_std_dev
            airport.mean_turnaround_time = airport_turnaround_times[airport.code]
            airport.turnaround_time_std_dev = (
                turnaround_time_variation * airport.mean_turnaround_time
            )
            airport.base_cancel_prob = airport_base_cancel_prob[airport.code]

            # Initialize the available aircraft list
            airport.num_available_aircraft = airport_initial_available_aircraft[
                airport.code
            ]
            i = 0
            while i < airport.num_available_aircraft:
                airport.available_aircraft.append(torch.tensor(0.0, device=device))
                i += 1

        # assign parameter to source supernode
        state.source_supernode.runway_use_time_std_dev = runway_use_time_std_dev

        # Simulate the movement of aircraft within the system for a fixed period of time
        t = torch.tensor(0.0, device=device)
        while not state.complete:
            # Update the current time
            t += delta_t

            # print(t)
            # print(f'pending incoming flights: {len(state.pending_incoming_flights)}')
            # print(f' pending network flights: {len(state.network_state.pending_flights)}')
            # print(f'      in transit flights: {len(state.network_state.in_transit_flights)}')
            # print(f'        LGA runway queue: {len(state.network_state.airports["LGA"].runway_queue)}')
            # exit()

            # All parked aircraft that are ready to turnaround get serviced
            for airport in state.network_state.airports.values():
                airport.update_available_aircraft(t)

            # If the maximum time has elapsed, add lots of reserve aircraft at each
            # airport. This is artificial and only done to ensure that the simulation
            # terminates.
            if t >= max_t:
                # print(f"TIME'S UP! Adding reserve aircraft at time {t}")
                for airport in state.network_state.airports.values():
                    airport.num_available_aircraft = airport.num_available_aircraft + 1
                    airport.available_aircraft.append(t)

            # All flights that are able to depart get moved to the runway queue at their
            # origin airport
            (
                network_ready_to_depart_flights, network_ready_times,
                incoming_ready_to_depart_flights, incoming_ready_times
            ) = \
                state.pop_ready_to_depart_flights(
                    t, var_prefix
                )
            
            for flight, ready_time in zip(network_ready_to_depart_flights, network_ready_times):
                queue_entry = QueueEntry(flight=flight, queue_start_time=ready_time)
                state.network_state.airports[flight.origin].runway_queue.append(queue_entry)

            for flight, ready_time in zip(incoming_ready_to_depart_flights, incoming_ready_times):
                queue_entry = DepartureQueueEntry(flight=flight, ready_time=ready_time)
                state.source_supernode.departure_queues[flight.destination].append(queue_entry)

            # All flights that are using the runway get serviced
            for airport in state.network_state.airports.values():
                network_departed_flights, network_landing_flights = \
                    airport.update_runway_queue(
                        t, var_prefix
                    )
                
                # TODO: update departure things from source supernode
                incoming_departed_flights = \
                    state.source_supernode.update_departure_queue_for_destination(
                        t, airport.code, var_prefix 
                        # maybe also need some info from the previous step?
                    )

                # Departing flights get added to the in-transit list, while landed flights
                # get added to the completed list
                state.add_in_transit_flights(
                    network_departed_flights, 
                    incoming_departed_flights, 
                    travel_times, 
                    travel_time_variation, 
                    var_prefix
                )
                state.add_completed_flights(network_landing_flights)

            # All flights that are in transit get moved to the runway queue at their
            # destination airport, if enough time has elapsed
            state.update_in_transit_flights(t)

        # print(f"---------- Completing day {day_ind} ----------")
        # print(f"# pending flights: {len(state.pending_flights)}")
        # print(f"# in-transit flights: {len(state.in_transit_flights)}")
        # print(f"# completed flights: {len(state.completed_flights)}")

        # Once we're done, return the state (this will include the actual arrival/departure
        # times for each aircraft)
        output_states.append(state)

    return output_states