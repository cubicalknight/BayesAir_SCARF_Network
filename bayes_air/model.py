"""Define a probabilistic model for an air traffic network."""
from copy import deepcopy

import pyro
import pyro.distributions as dist
import torch

from bayes_air.network import NetworkState, AugmentedNetworkState
from bayes_air.types import QueueEntry, DepartureQueueEntry, AirportCode, Time

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
        # To ignore cancellations, just provide practically infinite reserves
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
                (
                    departed_flights, landing_flights,
                ) = (
                    airport.update_runway_queue(t, var_prefix)
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
    delta_t: float = 0.1,
    max_t: float = FAR_FUTURE_TIME,
    device=None,

    travel_times_dict: dict[tuple[AirportCode, AirportCode], float] = None,

    obs_none: bool = False,
    verbose: bool = False,

    include_cancellations: bool = True,

    mean_service_time_effective_hrs: int = 6,
    mean_turnaround_time_effective_hrs: int = 6,
    base_cancel_prob_effective_hrs: int = 24,
    # travel_time_effective_hrs: int = 24,
    incoming_residual_delay_effective_hrs: int = 6,
    outgoing_residual_delay_effective_hrs: int = 6,
    effective_start_hr: int = 6,
    effective_end_hr: int = 24,

    # if actual times are chosen, then these are ignored
    source_use_actual_nas_delay: bool = False,
    source_use_actual_carrier_delay: bool = False,
    source_use_actual_weather_delay: bool = False,
    source_use_actual_security_delay: bool = False,
    source_use_actual_late_aircraft_delay: bool = False,
    # only one of below can be set, deptime overrides wo
    source_use_actual_departure_time: bool = False,
    source_use_actual_wheels_off_time: bool = False,

    source_use_actual_cancelled: bool = True,

    model_incoming_residual_departure_delay: bool = False,
    model_outgoing_residual_departure_delay: bool = False,

    use_failure_prior: bool = False,
    use_nominal_prior: bool = False,

    # # mostly for debugging
    # network_use_actual_nas_delay: bool = False,
    # network_use_actual_carrier_delay: bool = False,
    # network_use_actual_weather_delay: bool = False,
    # network_use_actual_security_delay: bool = False,
    # network_use_actual_late_aircraft_delay: bool = False,
    # network_use_actual_departure_time: bool = False,
    # network_use_actual_wheels_off_time: bool = False,

    soft_max_holding_time: float = None,
    max_holding_time: float = None,

    max_waiting_time: float = None,

    do_mle: bool = False
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

    # setup latent parameter effective time splits

    total_effective_hrs = effective_end_hr - effective_start_hr
    num_mean_service_time = max(
        1, total_effective_hrs // mean_service_time_effective_hrs
    )
    num_mean_turnaround_time = max(
        1, total_effective_hrs // mean_turnaround_time_effective_hrs
    )
    num_base_cancel_prob = max(
        1, total_effective_hrs // base_cancel_prob_effective_hrs
    )
    # num_travel_time = max(
    #     1, total_effective_hrs // travel_time_effective_hrs
    # )
    num_incoming_residual_delay = max(
        1, total_effective_hrs // incoming_residual_delay_effective_hrs
    )
    num_outgoing_residual_delay = max(
        1, total_effective_hrs // outgoing_residual_delay_effective_hrs
    )

    def t_to_t_idx(t, effective_hrs, num_idx):
        return max(0, min(
            int((t - effective_start_hr) // effective_hrs),
            num_idx - 1 # i think the .item() is unneeded
        ))

    # Define system-level parameters
    runway_use_time_std_dev = pyro.param(
        "runway_use_time_std_dev",
        torch.tensor(0.001, device=device),  # used to be 0.025
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
        t_idx: {
            # code: pyro.sample(
            #     f"{code}_{t_idx}_mean_turnaround_time",
            #     dist.Gamma(
            #         torch.tensor(1.0, device=device), 
            #         torch.tensor(2.0, device=device)
            #     ).mask(not do_mle),
            # )
            code: torch.tensor(0.5, device=device) # debugging
            for code in network_airport_codes
        }
        for t_idx in range(num_mean_turnaround_time)
    }

    def _gamma_dist_from_mean_std(mean, std):
        # std**2 = shape/rate**2
        # mean = shape/rate
        shape = (mean/std)**2
        rate = mean/std**2
        return dist.Gamma(
            torch.tensor(shape, device=device),
            torch.tensor(rate, device=device)
        )
    
    def _beta_dist_from_mean_std(mean, std):
        # α = μν, β = (1 − μ)ν
        alpha = mean * std**2
        beta = (1-mean) * std**2
        return dist.Beta(
            torch.tensor(alpha, device=device),
            torch.tensor(beta, device=device)
        )

    if use_nominal_prior:
        mst_dist = _gamma_dist_from_mean_std(0.0125, 0.01)
    elif use_failure_prior:
        mst_dist = _gamma_dist_from_mean_std(0.025, 0.015)
    else:
        mst_dist = dist.AffineBeta(
            torch.tensor(1.0, device=device),
            torch.tensor(1.0, device=device),
            loc=torch.tensor(.010).to(device),
            scale=torch.tensor(.020).to(device),
        )

    airport_service_times = {
        t_idx: {
            code: pyro.sample(
                f"{code}_{t_idx}_mean_service_time",
                mst_dist.mask(not do_mle)
            )
            for code in network_airport_codes
        }
        for t_idx in range(num_mean_service_time)
    }

    # with LGA only this is empty...
    network_travel_times = {
        # (origin, destination): pyro.sample(
        #     f"travel_time_{origin}_{destination}",
        #     dist.Gamma(
        #         torch.tensor(4.0, device=device), 
        #         torch.tensor(1.25, device=device)
        #     ).mask(not do_mle),
        # )
        # for origin in network_airport_codes
        # for destination in network_airport_codes
        # if origin != destination
    }

    if include_cancellations:
        airport_initial_available_aircraft = {
            code: torch.exp(
                # pyro.sample(
                #     f"{code}_log_initial_available_aircraft",
                #     dist.Normal(
                #         torch.tensor(1.0, device=device),
                #         torch.tensor(1.0, device=device),
                #     ).mask(not do_mle),
                # )
                torch.tensor(10.0, device=device) # debugging
            )
            for code in network_airport_codes
        }
        airport_base_cancel_prob = {
            t_idx: {
                code: torch.exp(
                    # -pyro.sample(
                    #     f"{code}_{t_idx}_base_cancel_neg_logprob",
                    #     _gamma_dist_from_mean_std(3.0, 1.0).mask(not do_mle)
                    # ) # note the negative
                    torch.tensor(-3.0, device=device) # testing
                )
                for code in network_airport_codes
            }
            for t_idx in range(num_base_cancel_prob)
        }
    else:
        # To ignore cancellations, just provide practically infinite reserves
        airport_initial_available_aircraft = {
            code: torch.tensor(1000.0, device=device) 
            for code in network_airport_codes
        }
        airport_base_cancel_prob = {
            t_idx: {
                code: torch.tensor(0.0, device=device) 
                for code in network_airport_codes
            }
            for t_idx in range(num_base_cancel_prob)
        }

    # sample residual delays, if modeling them
    if model_incoming_residual_departure_delay:
        incoming_residual_departure_delay = {
            t_idx: {
                # code: pyro.sample(
                #     f"{code}_{t_idx}_residual_departure_delay",
                #     _gamma_dist_from_mean_std(.5, 1).mask(not do_mle)
                # )
                code: torch.tensor(0.0).to(device)
                for code in network_airport_codes
            }
            for t_idx in range(num_incoming_residual_delay)
        }

    # sample latent variables for variables outside network
    incoming_airport_codes = set()
    for state in states:
        current_codes = state.source_supernode.source_codes
        incoming_airport_codes.update(current_codes)
    incoming_airport_codes = list(incoming_airport_codes)

    # trying something
    incoming_travel_times = {
        (origin, destination): 
            # pyro.sample(
            #     f"travel_time_{origin}_{destination}",
            #     _gamma_dist_from_mean_std(
            #         travel_times_dict[(origin, destination)], .5
            #     ).mask(not do_mle)
            #     if travel_times_dict is not None else
            #     _gamma_dist_from_mean_std(2.0, 1.5).mask(not do_mle)
            # )
            travel_times_dict[(origin, destination)]
        for origin in incoming_airport_codes
        for destination in network_airport_codes
    }

    travel_times = network_travel_times | incoming_travel_times


    # things for handling max queue waiting times for arr/dep flights

    use_max_holding_time = (max_holding_time is not None)
    use_max_waiting_time = (max_waiting_time is not None)

    airport_use_max_holding_times = {
        code: use_max_holding_time 
        for code in network_airport_codes
    }
    if use_max_holding_time:
        airport_soft_max_holding_times = {
            code: (
                pyro.param(
                    f"{code}_soft_max_holding_time",
                    torch.tensor(soft_max_holding_time, device=device),
                    dist.constraints.nonnegative
                )
                # if soft_max_holding_time is not None
                # else pyro.sample(
                #     f"{code}_soft_max_holding_time",
                #     dist.Uniform(
                #         torch.tensor(0.0, device=device),
                #         torch.tensor(max_holding_time, device=device)
                #     ).mask(not do_mle)
                # )
            )
            for code in network_airport_codes
        }
        airport_max_holding_times = {
            code: pyro.param(
                f"{code}_max_holding_time",
                torch.tensor(max_holding_time, device=device),
                dist.constraints.nonnegative
            )
            for code in network_airport_codes
        }

    airport_use_max_waiting_times = {
        code: use_max_waiting_time 
        for code in network_airport_codes
    }
    if use_max_waiting_time:
        airport_max_waiting_times = {
            code: pyro.param(
                f"{code}_max_waiting_time",
                torch.tensor(max_waiting_time, device=device),
                dist.constraints.nonnegative
            )
            for code in network_airport_codes
        }

    # Simulate for each state
    output_states = []
    # for day_ind in pyro.plate("days", len(states)):
    for day_ind in pyro.markov(range(len(states)), history=1):
        state = states[day_ind]
        # var_prefix = f"day{day_ind}_"
        var_prefix = f"{state.day_str}_"

        # set adjusted departure time (for incoming flights)
        for flight in state.pending_incoming_flights:

            if source_use_actual_departure_time:
                adj_time = flight.actual_departure_time
            elif source_use_actual_wheels_off_time:
                adj_time = flight.wheels_off_time

            else:
                adj_time = flight.scheduled_departure_time
                if source_use_actual_carrier_delay:
                    adj_time += flight.carrier_delay
                if source_use_actual_late_aircraft_delay:
                    adj_time += flight.late_aircraft_delay
                if source_use_actual_nas_delay:
                    adj_time += flight.nas_delay
                if source_use_actual_weather_delay:
                    adj_time += flight.weather_delay
                if source_use_actual_security_delay:
                    adj_time += flight.security_delay

            flight.adjusted_departure_time = (
                adj_time if adj_time is not None
                else flight.scheduled_departure_time
            )

            flight.adjusted_cancelled = (
                flight.cancelled if source_use_actual_cancelled
                else False
            )

        state.pending_incoming_flights.sort(
            key=lambda x: x.adjusted_departure_time
        )

        # print(f"============= Starting day {day_ind} =============")
        # print(f"# pending flights: {len(state.pending_flights)}")
        # print(f"Initial aircraft: {airport_initial_available_aircraft}")
        # print(f"# in-transit flights: {len(state.in_transit_flights)}")
        # print(f"# completed flights: {len(state.completed_flights)}")
        # print("Travel times:")
        # print(travel_times)

        # Assign the latent variables to the airports
        for airport in state.network_state.airports.values():
            # airport.mean_service_time = airport_service_times[0][airport.code]
            airport.runway_use_time_std_dev = runway_use_time_std_dev
            # airport.mean_turnaround_time = airport_turnaround_times[0][airport.code]
            airport.turnaround_time_std_dev = (
                turnaround_time_variation * airport.mean_turnaround_time
            )
            # airport.base_cancel_prob = airport_base_cancel_prob[0][airport.code]

            # Initialize the available aircraft list
            airport.num_available_aircraft = airport_initial_available_aircraft[
                airport.code
            ]
            i = 0
            while i < airport.num_available_aircraft:
                airport.available_aircraft.append(torch.tensor(0.0, device=device))
                i += 1

            airport.use_max_holding_time = airport_use_max_holding_times[airport.code]
            airport.use_max_waiting_time = airport_use_max_waiting_times[airport.code]

            if airport.use_max_holding_time:
                airport.max_holding_time = airport_max_holding_times[airport.code]
                airport.soft_max_holding_time = airport_soft_max_holding_times[airport.code]

            if airport.use_max_waiting_time:
                airport.max_waiting_time = airport_max_waiting_times[airport.code]

        # assign parameter to source supernode
        state.source_supernode.runway_use_time_std_dev = runway_use_time_std_dev

        # Simulate the movement of aircraft within the system for a fixed period of time
        t = torch.tensor(0.0, device=device)
        while not state.complete:
            # Update the current time
            t += delta_t

            # update parameters as necessary
            for airport in state.network_state.airports.values():
                # after 24h, just use last one
                mst_idx = t_to_t_idx(
                    t, mean_service_time_effective_hrs, num_mean_service_time
                )
                mtt_idx = t_to_t_idx(
                    t, mean_turnaround_time_effective_hrs, num_mean_turnaround_time
                )
                bcp_idx = t_to_t_idx(
                    t, base_cancel_prob_effective_hrs, num_base_cancel_prob
                )
                # print(mst_idx, mtt_idx, bcp_idx)
                airport.mean_service_time = airport_service_times[mst_idx][airport.code]
                airport.mean_turnaround_time = airport_turnaround_times[mtt_idx][airport.code]
                airport.base_cancel_prob = airport_base_cancel_prob[bcp_idx][airport.code]

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
                (
                    network_departed_flights, network_landing_flights,
                    network_cancelled_flights, network_diverted_flights,
                ) = (
                    airport.update_runway_queue(t, var_prefix)
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

                state.add_completed_flights(network_cancelled_flights)
                state.add_completed_flights(network_diverted_flights)

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






def augmented_air_traffic_network_model_simplified(
    states: list[AugmentedNetworkState],
    delta_t: float = 0.1,
    max_t: float = FAR_FUTURE_TIME,
    device=None,

    travel_times_dict: dict[tuple[AirportCode, AirportCode], float] = None,
    initial_aircraft = 10.0,

    obs_none: bool = False,
    verbose: bool = False,

    include_cancellations: bool = True,

    mean_service_time_effective_hrs: int = 24,
    mean_turnaround_time_effective_hrs: int = 24,
    base_cancel_prob_effective_hrs: int = 24,
    # travel_time_effective_hrs: int = 24,
    effective_start_hr: int = 6,
    effective_end_hr: int = 24,

    # if actual times are chosen, then these are ignored
    source_use_actual_nas_delay: bool = False,
    source_use_actual_carrier_delay: bool = False,
    source_use_actual_weather_delay: bool = False,
    source_use_actual_security_delay: bool = False,
    source_use_actual_late_aircraft_delay: bool = False,
    # only one of below can be set, deptime overrides wo
    source_use_actual_departure_time: bool = False,
    source_use_actual_wheels_off_time: bool = False,

    source_use_actual_cancelled: bool = True,

    soft_max_holding_time: float = None,
    max_holding_time: float = None,
    max_waiting_time: float = None,

    mst_prior: dist.Distribution = None,

    do_mle: bool = False
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

    # setup latent parameter effective time splits

    total_effective_hrs = effective_end_hr - effective_start_hr
    num_mean_service_time = max(
        1, total_effective_hrs // mean_service_time_effective_hrs
    )
    num_mean_turnaround_time = max(
        1, total_effective_hrs // mean_turnaround_time_effective_hrs
    )
    num_base_cancel_prob = max(
        1, total_effective_hrs // base_cancel_prob_effective_hrs
    )
    # num_travel_time = max(
    #     1, total_effective_hrs // travel_time_effective_hrs
    # )

    def t_to_t_idx(t, effective_hrs, num_idx):
        return max(0, min(
            int((t - effective_start_hr) // effective_hrs),
            num_idx - 1 # i think the .item() is unneeded
        ))

    # Define system-level parameters
    runway_use_time_std_dev = pyro.param(
        "runway_use_time_std_dev",
        torch.tensor(0.001, device=device),  # used to be 0.025
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

    def _gamma_dist_from_mean_std(mean, std):
        # std**2 = shape/rate**2
        # mean = shape/rate
        shape = (mean/std)**2
        rate = mean/std**2
        return dist.Gamma(
            torch.tensor(shape, device=device),
            torch.tensor(rate, device=device)
        )
    
    def _beta_dist_from_mean_std(mean, std):
        # α = μν, β = (1 − μ)ν
        alpha = mean * std**2
        beta = (1-mean) * std**2
        return dist.Beta(
            torch.tensor(alpha, device=device),
            torch.tensor(beta, device=device)
        )
    
    def _affine_beta_dist_from_mean_std(mean, std, loc, scale):
        raise NotImplementedError

    airport_turnaround_times = {
        t_idx: {
            # code: pyro.sample(
            #     f"{code}_{t_idx}_mean_turnaround_time",
            #     dist.Gamma(
            #         torch.tensor(1.0, device=device), 
            #         torch.tensor(2.0, device=device)
            #     ).mask(not do_mle),
            # )
            code: torch.tensor(0.5, device=device) # debugging
            for code in network_airport_codes
        }
        for t_idx in range(num_mean_turnaround_time)
    }

    if mst_prior is None:
        mst_dist = dist.AffineBeta(
            torch.tensor(1.0, device=device),
            torch.tensor(1.0, device=device),
            loc=torch.tensor(.010).to(device),
            scale=torch.tensor(.020).to(device),
        )
    else:
        mst_dist = mst_prior

    airport_service_times = {
        t_idx: {
            code: pyro.sample(
                f"{code}_{t_idx}_mean_service_time",
                mst_dist.mask(not do_mle)
            )
            for code in network_airport_codes
        }
        for t_idx in range(num_mean_service_time)
    }

    # with LGA only this is empty...
    network_travel_times = {}

    if include_cancellations:
        airport_initial_available_aircraft = {
            code: torch.exp(
                torch.tensor(
                    initial_aircraft,
                    device=device
                ) # default i guess
            )
            for code in network_airport_codes
        }
        airport_base_cancel_prob = {
            t_idx: {
                code: torch.exp(
                    # -pyro.sample(
                    #     f"{code}_{t_idx}_base_cancel_neg_logprob",
                    #     _gamma_dist_from_mean_std(3.0, 1.0).mask(not do_mle)
                    # ) # note the negative
                    torch.tensor(-3.0, device=device) # testing
                )
                for code in network_airport_codes
            }
            for t_idx in range(num_base_cancel_prob)
        }
    else:
        # To ignore cancellations, just provide practically infinite reserves
        airport_initial_available_aircraft = {
            code: torch.tensor(1000.0, device=device) 
            for code in network_airport_codes
        }
        airport_base_cancel_prob = {
            t_idx: {
                code: torch.tensor(0.0, device=device) 
                for code in network_airport_codes
            }
            for t_idx in range(num_base_cancel_prob)
        }

    # sample latent variables for variables outside network
    incoming_airport_codes = set()
    for state in states:
        current_codes = state.source_supernode.source_codes
        incoming_airport_codes.update(current_codes)
    incoming_airport_codes = list(incoming_airport_codes)

    # trying something
    incoming_travel_times = {
        (origin, destination): 
            travel_times_dict[(origin, destination)]
        for origin in incoming_airport_codes
        for destination in network_airport_codes
    }

    travel_times = network_travel_times | incoming_travel_times


    # things for handling max queue waiting times for arr/dep flights

    use_max_holding_time = (max_holding_time is not None)
    use_max_waiting_time = (max_waiting_time is not None)

    airport_use_max_holding_times = {
        code: use_max_holding_time 
        for code in network_airport_codes
    }
    if use_max_holding_time:
        airport_soft_max_holding_times = {
            code: (
                pyro.param(
                    f"{code}_soft_max_holding_time",
                    torch.tensor(soft_max_holding_time, device=device),
                    dist.constraints.nonnegative
                )
                # if soft_max_holding_time is not None
                # else pyro.sample(
                #     f"{code}_soft_max_holding_time",
                #     dist.Uniform(
                #         torch.tensor(0.0, device=device),
                #         torch.tensor(max_holding_time, device=device)
                #     ).mask(not do_mle)
                # )
            )
            for code in network_airport_codes
        }
        airport_max_holding_times = {
            code: pyro.param(
                f"{code}_max_holding_time",
                torch.tensor(max_holding_time, device=device),
                dist.constraints.nonnegative
            )
            for code in network_airport_codes
        }

    airport_use_max_waiting_times = {
        code: use_max_waiting_time 
        for code in network_airport_codes
    }
    if use_max_waiting_time:
        airport_max_waiting_times = {
            code: pyro.param(
                f"{code}_max_waiting_time",
                torch.tensor(max_waiting_time, device=device),
                dist.constraints.nonnegative
            )
            for code in network_airport_codes
        }

    # Simulate for each state
    output_states = []
    # for day_ind in pyro.plate("days", len(states)):
    for day_ind in pyro.markov(range(len(states)), history=1):
        state = states[day_ind]
        # var_prefix = f"day{day_ind}_"
        var_prefix = f"{state.day_str}_"

        # set adjusted departure time (for incoming flights)
        for flight in state.pending_incoming_flights:

            if source_use_actual_departure_time:
                adj_time = flight.actual_departure_time
            elif source_use_actual_wheels_off_time:
                adj_time = flight.wheels_off_time

            else:
                adj_time = flight.scheduled_departure_time
                if source_use_actual_carrier_delay:
                    adj_time += flight.carrier_delay
                if source_use_actual_late_aircraft_delay:
                    adj_time += flight.late_aircraft_delay
                if source_use_actual_nas_delay:
                    adj_time += flight.nas_delay
                if source_use_actual_weather_delay:
                    adj_time += flight.weather_delay
                if source_use_actual_security_delay:
                    adj_time += flight.security_delay

            flight.adjusted_departure_time = (
                adj_time if adj_time is not None
                else flight.scheduled_departure_time
            )

            flight.adjusted_cancelled = (
                flight.cancelled if source_use_actual_cancelled
                else False
            )

        state.pending_incoming_flights.sort(
            key=lambda x: x.adjusted_departure_time
        )

        # print(f"============= Starting day {day_ind} =============")
        # print(f"# pending flights: {len(state.pending_flights)}")
        # print(f"Initial aircraft: {airport_initial_available_aircraft}")
        # print(f"# in-transit flights: {len(state.in_transit_flights)}")
        # print(f"# completed flights: {len(state.completed_flights)}")
        # print("Travel times:")
        # print(travel_times)

        # Assign the latent variables to the airports
        for airport in state.network_state.airports.values():
            # airport.mean_service_time = airport_service_times[0][airport.code]
            airport.runway_use_time_std_dev = runway_use_time_std_dev
            # airport.mean_turnaround_time = airport_turnaround_times[0][airport.code]
            airport.turnaround_time_std_dev = (
                turnaround_time_variation * airport.mean_turnaround_time
            )
            # airport.base_cancel_prob = airport_base_cancel_prob[0][airport.code]

            # Initialize the available aircraft list
            airport.num_available_aircraft = airport_initial_available_aircraft[
                airport.code
            ]
            i = 0
            while i < airport.num_available_aircraft:
                airport.available_aircraft.append(torch.tensor(0.0, device=device))
                i += 1

            airport.use_max_holding_time = airport_use_max_holding_times[airport.code]
            airport.use_max_waiting_time = airport_use_max_waiting_times[airport.code]

            if airport.use_max_holding_time:
                airport.max_holding_time = airport_max_holding_times[airport.code]
                airport.soft_max_holding_time = airport_soft_max_holding_times[airport.code]

            if airport.use_max_waiting_time:
                airport.max_waiting_time = airport_max_waiting_times[airport.code]

        # assign parameter to source supernode
        state.source_supernode.runway_use_time_std_dev = runway_use_time_std_dev

        # Simulate the movement of aircraft within the system for a fixed period of time
        t = torch.tensor(0.0, device=device)
        while not state.complete:
            # Update the current time
            t += delta_t

            # update parameters as necessary
            for airport in state.network_state.airports.values():
                # after 24h, just use last one
                mst_idx = t_to_t_idx(
                    t, mean_service_time_effective_hrs, num_mean_service_time
                )
                mtt_idx = t_to_t_idx(
                    t, mean_turnaround_time_effective_hrs, num_mean_turnaround_time
                )
                bcp_idx = t_to_t_idx(
                    t, base_cancel_prob_effective_hrs, num_base_cancel_prob
                )
                # print(mst_idx, mtt_idx, bcp_idx)
                airport.mean_service_time = airport_service_times[mst_idx][airport.code]
                airport.mean_turnaround_time = airport_turnaround_times[mtt_idx][airport.code]
                airport.base_cancel_prob = airport_base_cancel_prob[bcp_idx][airport.code]

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
                (
                    network_departed_flights, network_landing_flights,
                    network_cancelled_flights, network_diverted_flights,
                ) = (
                    airport.update_runway_queue(t, var_prefix)
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

                state.add_completed_flights(network_cancelled_flights)
                state.add_completed_flights(network_diverted_flights)

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