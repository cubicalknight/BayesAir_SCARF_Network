"""Define methods for working with schedules."""
import pandas as pd
import torch
from pathlib import Path

from bayes_air.types import Airport, Flight, Time, Schedule, AirportCode, SourceSupernode

# Parse the provided data into our custom data structures
def parse_flight(schedule_row: tuple, device=None) -> Flight:
    """
    Parse a row of the schedule into an Flight object.

    Args:
        schedule_row: a tuple of the following items
            - flight_number
            - origin_airport
            - destination_airport
            - cancelled
            - scheduled_departure_time
            - scheduled_arrival_time
            - actual_departure_time
            - actual_arrival_time
            - wheels_off_time
            - wheels_on_time
        device: The device to use for the tensors
    """
    if device is None:
        device = torch.device("cpu")

    flight_number = schedule_row["flight_number"]
    origin_airport = schedule_row["origin_airport"]
    destination_airport = schedule_row["destination_airport"]
    scheduled_departure_time = schedule_row["scheduled_departure_time"]
    scheduled_arrival_time = schedule_row["scheduled_arrival_time"]
    actual_departure_time = schedule_row["actual_departure_time"]
    actual_arrival_time = schedule_row["actual_arrival_time"]
    wheels_off_time = schedule_row["wheels_off_time"]
    wheels_on_time = schedule_row["wheels_on_time"]
    is_incoming_flight = schedule_row["is_incoming_flight"]
    is_outgoing_flight = schedule_row["is_outgoing_flight"]
    cancelled = (
        torch.tensor(1.0, device=device)
        if schedule_row["cancelled"]
        else torch.tensor(0.0, device=device)
    )

    # If the flight was cancelled, set the measured times to None
    if schedule_row["cancelled"]:
        actual_departure_time = None
        actual_arrival_time = None
        wheels_off_time = None
        wheels_on_time = None

    return Flight(
        flight_number=flight_number,
        origin=origin_airport,
        destination=destination_airport,
        scheduled_departure_time=Time(scheduled_departure_time).to(device),
        scheduled_arrival_time=Time(scheduled_arrival_time).to(device),
        actually_cancelled=cancelled,
        actual_departure_time=Time(actual_departure_time).to(device)
        if actual_departure_time is not None
        else None,
        actual_arrival_time=Time(actual_arrival_time).to(device)
        if actual_arrival_time is not None
        else None,
        wheels_off_time=Time(wheels_off_time).to(device)
        if wheels_off_time is not None
        else None,
        wheels_on_time=Time(wheels_on_time).to(device)
        if wheels_on_time is not None
        else None,
        is_incoming_flight=is_incoming_flight,
        is_outgoing_flight=is_outgoing_flight,
    )


def parse_schedule(
    schedule_df: Schedule, device=None
) -> tuple[list[Flight], list[Flight]]:
    """Parse a pandas dataframe for a schedule into a list of pending flights.

    Args:
        schedule_df: A pandas dataframe with the following columns:
            flight_number: The flight number
            origin_airport: The airport code of the origin airport
            destination_airport: The airport code of the destination airport
            scheduled_departure_time: The scheduled departure time
            scheduled_arrival_time: The scheduled arrival time
            actual_departure_time: The actual departure time
            actual_arrival_time: The actual arrival time
            wheels_off_time: The time the wheels left the ground
            wheels_on_time: The time the wheels touched the ground
        device: The device to use for the tensors

    Returns:
        a list of flights, and
        a list of airports
    """
    # Get a list of flights
    flights = [parse_flight(row, device=device) for _, row in schedule_df.iterrows()]

    # Get a list of unique airport codes from the origin and destination columns
    airport_codes = pd.concat(
        [schedule_df["origin_airport"], schedule_df["destination_airport"]]
    ).unique()
    # Create an airport object for each airport code
    airports = [Airport(code) for code in airport_codes]

    return flights, airports


def split_full_schedule(
    schedule_df: Schedule,
    network_airport_codes: list[AirportCode]
) -> tuple[Schedule, Schedule]:
    """
    takes in schedule, splits into incoming and network flights,
    where incoming is originating outside the network

    Args:
        schedule_df:
        network_aiport_codes:

    returns:
        network_schedule_df, incoming_schedule_df
    """

    network_mask = schedule_df["origin_airport"].isin(network_airport_codes)
    incoming_mask = ~network_mask

    network_schedule_df = schedule_df[network_mask].copy()
    incoming_schedule_df = schedule_df[incoming_mask].copy()

    network_schedule_df["is_incoming_flight"] = False
    network_schedule_df["is_outgoing_flight"] = ~(
        network_schedule_df["destination_airport"].isin(network_airport_codes)
    )
    
    incoming_schedule_df["is_incoming_flight"] = True
    incoming_schedule_df["is_outgoing_flight"] = False

    return network_schedule_df, incoming_schedule_df


def parse_split_schedule(
    network_schedule_df: Schedule, 
    incoming_schedule_df: Schedule,
    device=None
) -> tuple[
    list[Flight], list[Flight], 
    list[Flight], list[Flight]
]:
    """Parse a pandas dataframe for a schedule into a list of pending flights.

    Args:
        network_schedule_df: A pandas dataframe with the following columns:
            flight_number: The flight number
            origin_airport: The airport code of the origin airport
            destination_airport: The airport code of the destination airport
            scheduled_departure_time: The scheduled departure time
            scheduled_arrival_time: The scheduled arrival time
            actual_departure_time: The actual departure time
            actual_arrival_time: The actual arrival time
            wheels_off_time: The time the wheels left the ground
            wheels_on_time: The time the wheels touched the ground
        incoming_schedule_df: same as above
        device: The device to use for the tensors

    Returns:
        a list of flights, and
        a list of airports, for both incoming and network
    """
    # print(network_schedule_df.dtypes, incoming_schedule_df.dtypes)

    # Get a list of flights
    network_flights = [
        parse_flight(row, device=device) 
        for _, row in network_schedule_df.iterrows()
    ]

    incoming_flights = [
        parse_flight(row, device=device) 
        for _, row in incoming_schedule_df.iterrows()
    ]

    # Get a list of unique airport codes for flight origins
    network_airport_codes = network_schedule_df["origin_airport"].unique()
    source_airport_codes = incoming_schedule_df["origin_airport"].unique()

    # Create an airport object for each airport code in network
    network_airports = [Airport(code) for code in network_airport_codes]
    source_supernode = SourceSupernode(
        source_codes=source_airport_codes,
        dest_codes=network_airport_codes,
    )

    return (
        network_flights, network_airports,
        incoming_flights, source_supernode,
    )
    

def split_and_parse_full_schedule(
    schedule_df: Schedule,
    network_airport_codes: list[AirportCode],
    device=None
) -> tuple[
    list[Flight], list[Flight], 
    list[Flight], list[Flight]
]:
    network_schedule_df, incoming_schedule_df = \
        split_full_schedule(schedule_df, network_airport_codes)
    
    return parse_split_schedule(
        network_schedule_df, incoming_schedule_df, device
    )
