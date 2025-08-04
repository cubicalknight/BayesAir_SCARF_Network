"""Define convenience types."""
import torch
import pandas as pd

# Convenient type aliases
AirportCode = str
Time = torch.tensor
Schedule = pd.DataFrame

CoreAirports = {
        'ATL': 'Hartsfield-Jackson Atlanta Intl',
        'BOS': 'Boston Logan Intl',
        'BWI': 'Baltimore/Washington Intl',
        'CLT': 'Charlotte Douglas Intl',
        'DCA': 'Ronald Reagan Washington National',
        'DEN': 'Denver Intl',
        'DFW': ' Dallas/Fort Worth Intl',
        'DTW': 'Detroit Metropolitan Wayne County',
        'EWR': 'Newark Liberty Intl',
        'FLL': 'Fort Lauderdale/Hollywood Intl',
        # 'HNL': 'Honolulu Intl',
        'IAD': 'Washington Dulles Intl',
        'IAH': 'George Bush Houston Intercontinental',
        'JFK': 'New York John F. Kennedy Intl',
        'LAS': 'Las Vegas McCarran Intl',
        'LAX': 'Los Angeles Intl',
        'LGA': 'New York LaGuardia',
        'MCO': 'Orlando Intl',
        'MDW': 'Chicago Midway',
        'MEM': 'Memphis Intl',
        'MIA': 'Miami Intl',
        'MSP': 'Minneapolis/St. Paul Intl',
        'ORD': 'Chicago O`Hare Intl',
        'PHL': 'Philadelphia Intl',
        'PHX': 'Phoenix Sky Harbor Intl',
        'SAN': 'San Diego Intl',
        'SEA': 'Seattle/Tacoma Intl',
        'SFO': 'San Francisco Intl',
        'SLC': 'Salt Lake City Intl',
        'TPA': 'Tampa Intl'
    }