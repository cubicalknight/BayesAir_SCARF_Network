# %%
import polars as pl
# import numpy as np

import sys
from pathlib import Path
import os
from tqdm import tqdm

import airportsdata as apd
from calendar import monthrange
import functools

from bayes_air.types.util import CoreAirports
# import networkx as nx
# import matplotlib.pyplot as plt


def save_to_parquet_and_csv(df, base_filename_csv, base_filename_parquet):
    """
    Saves the DataFrame to both Parquet and CSV formats.
    Args:
        df (polars.DataFrame): The DataFrame to save.
        base_filename (str): The base filename (without extension) to use for the output files.
    """
    df.write_parquet(f"{base_filename_parquet}.parquet")
    df.write_csv(f"{base_filename_csv}.csv")


# TODO Before full network evaluation combine into single query
def filter_for_airport(raw_csv_path, airport_code, out_dir):
    """    Filters a CSV file for rows that match a specific airport code.
    Args:
        csv_path (str): Path to the CSV file.
        airport_code (str): The airport code to filter by.
    Returns:
        polars.DataFrame: A DataFrame containing only the rows that match the airport code.
    """
    if not os.path.exists(raw_csv_path):
        raise FileNotFoundError(f"The file {raw_csv_path} does not exist.")
    
    raw_csv_path = Path(raw_csv_path).resolve()

    delay_cols = ['ArrDelay', 'DepDelay']
    split_delay_cols = ['CarrierDelay', 'WeatherDelay', 'NASDelay', 'SecurityDelay', 'LateAircraftDelay']
    all_delay_cols = delay_cols + split_delay_cols
    scheduled_cols = ['CRSDepTime', 'CRSArrTime']
    actual_dep_cols = ['DepTime', 'WheelsOff']
    actual_arr_cols = ['ArrTime', 'WheelsOn']
    actual_cols = actual_arr_cols + actual_dep_cols
    time_cols = scheduled_cols + actual_cols

    df = pl.read_csv(raw_csv_path)
    df = df.filter((pl.col("Origin") == airport_code) | (pl.col("Dest") == airport_code))

    int_cols_to_convert = []
    for col, dtype in df.schema.items():
        if dtype == pl.Int64 and "ID" not in col and col != 'Flight_Number_Reporting_Airline':
            int_cols_to_convert.append(col)
    
    # display(df)
    # show all cols
    # print(df.columns)
    print(f"Converting {len(int_cols_to_convert)} columns to Int16")
    # display(df)
    # print(int_cols_to_convert)
    df = df.with_columns(
        [pl.col(col).cast(pl.Int16).alias(col) for col in int_cols_to_convert]
        + [
            pl.col('Cancelled').cast(pl.Boolean),
            pl.col('Diverted').cast(pl.Boolean),
            pl.col('Flight_Number_Reporting_Airline').cast(pl.String),
            pl.col('FlightDate').str.strptime(pl.Datetime, format="%Y-%m-%d"),  # Adjust format as needed
        ]
    )

    if 'ArrTime' not in int_cols_to_convert and 'DepTime' not in int_cols_to_convert:
        # Convert time columns to Int64 if not already converted
        df = df.with_columns(
            pl.when(pl.col('ArrTime') == "").then(None).otherwise(pl.col('ArrTime')).cast(pl.Int16).alias('ArrTime'),
            pl.when(pl.col('DepTime') == "").then(None).otherwise(pl.col('DepTime')).cast(pl.Int16).alias('DepTime')
        )

    # sys.exit(0)  # Exit after reading the CSV to avoid further processing in this script

    # Filter delays  
    df = df.with_columns(
        [pl.when((pl.col('ArrDelay') < 15) | (pl.col("ArrTime") == pl.col("CRSArrTime"))).then(0).otherwise(pl.col(col)).alias(col) for col in split_delay_cols]
    )

    # Handle diversions and cancellations
    df = df.with_columns(
        [pl.when((pl.col('Diverted') != 0) | (pl.col('Cancelled') != 0)).then(9999).otherwise(pl.col(col)).alias(col) for col in all_delay_cols]
    )

    df = df.with_columns(
        [pl.when(pl.col('Diverted').is_null()).then(9999).otherwise(pl.col(col)).alias(col) for col in actual_arr_cols]
    )

    df = df.with_columns(
        [pl.when(pl.col('Cancelled').is_null()).then(9999).otherwise(pl.col(col)).alias(col) for col in actual_cols]
    )


    # Set CancellationCode to 'Z when not cancelled
    df = df.with_columns(
        pl.when((pl.col('Cancelled') == 0)).then(pl.lit('Z')).otherwise(pl.col('CancellationCode')).alias('CancellationCode')
    )

    string_cols = [
        'Origin', 'Dest', 
        'Tail_Number', 'Flight_Number_Reporting_Airline', 
        'Reporting_Airline', 'CancellationCode'
    ]
    df = df.with_columns([
        pl.when(pl.col(col) == '').then(None).otherwise(pl.col(col)).alias(col)
        for col in string_cols
    ])

    # Drop rows with missing values in key columns
    ba_cols = [
        'FlightDate', 'Origin', 'Dest', 'Flight_Number_Reporting_Airline', 
        'CRSDepTime', 'DepTime', 'CRSArrTime', 'ArrTime', 
        'WheelsOff', 'WheelsOn', 'Cancelled'
    ]
    df = df.drop_nulls(subset=ba_cols)

    # Make categorical columns
    df = df.with_columns([
        pl.col('Reporting_Airline').cast(pl.Categorical),
        pl.col('CancellationCode').cast(pl.Categorical)
    ])

    # Save to Parquet and CSV
    out_dir_csv = Path(out_dir) / 'csv'
    out_dir_parquet = Path(out_dir) / 'parquet'

    year = df['Year'].unique().item()

    year_out_dir_csv = out_dir_csv / f'{airport_code}' / f'{year:04d}' 
    year_out_dir_parquet = out_dir_parquet / f'{airport_code}' / f'{year:04d}'
    year_out_dir_csv.mkdir(parents=True, exist_ok=True)
    year_out_dir_parquet.mkdir(parents=True, exist_ok=True)

    save_path_csv = year_out_dir_csv / (raw_csv_path.stem + f"_{airport_code}")
    save_path_parquet = year_out_dir_parquet / (raw_csv_path.stem + f"_{airport_code}")
    print(f"Saving processed data to {save_path_parquet}.parquet and {save_path_csv}.csv")
    save_to_parquet_and_csv(df, save_path_csv, save_path_parquet)

    return df, save_path_parquet, save_path_csv


# Only daily splits
def split_initial_work(data_path, out_dir):
                    #    , time_res):

    # if time_res == "yearly":
    #     split_monthly = False
    #     split_daily = False
    # elif time_res == "monthly":
    #     split_monthly = True
    #     split_daily = False
    # elif time_res == "daily":
    #     split_monthly = True
    #     split_daily = True
    # else:
    #     raise ValueError("time_res must be one of daily, monthly, yearly")

    data_path = Path(data_path).resolve()
    
    out_dir = data_path.parent / f'{data_path.stem}_daily' if out_dir is None else out_dir
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    year = int(data_path.stem.split('_')[9]) #.split('-')
    # start_year = int(years[0])
    # end_year = int(years[1])

    month = int(data_path.stem.split('_')[10])

    out_head = "_".join(data_path.stem.split('_')[:8])
    out_tail = data_path.stem.split('_')[-1]

    out_dir_parquet = out_dir / 'parquet' / out_tail
    out_dir_csv = out_dir / 'csv' / out_tail

    out_dir_csv.mkdir(parents=True, exist_ok=True)
    out_dir_parquet.mkdir(parents=True, exist_ok=True)

    df = pl.read_parquet(data_path)
    df_date = df['FlightDate'].dt

    # year_mask = {
    #     year: (df_date.year == year)
    #     for year in range(start_year, end_year+1)
    # }
    # month_mask = {
    #     month: (df_date.month == month)
    #     for month in range(1, 13)
    # }
    day_mask = {
        day: (df_date.day() == day)
        for day in range(1, 32)
    }

    return (
        data_path, out_dir,
        year, month,
        out_head, out_tail,
        out_dir_parquet, out_dir_csv,
        df,
        day_mask
    )
    # return (
    #     data_path, out_dir, 
    #     start_year, end_year, 
    #     out_head, out_tail, 
    #     out_dir_parquet, out_dir_csv, 
    #     df, year_mask, month_mask, day_mask,
    #     split_monthly, split_daily
    # )



# Need to revise to handle OT dataset since all data comes as monthly
# There is the IBM hosted dataset, but this is deprecated https://developer.ibm.com/data/airline/
def split_by_time_unified(data_path, time_res, out_dir=None):
    # (
    #     data_path, out_dir, 
    #     start_year, end_year, 
    #     out_head, out_tail, 
    #     out_dir_parquet, out_dir_csv, 
    #     df, year_mask, month_mask, day_mask,
    #     split_monthly, split_daily
    # ) = \
    #     split_initial_work(data_path, out_dir, time_res)

    (   data_path, out_dir,
        year, month,
        out_head, out_tail, 
        out_dir_parquet, out_dir_csv, 
        df, 
        day_mask
    ) = split_initial_work(data_path, out_dir)

    # for year in (pbar_year := tqdm(range(start_year, end_year+1), leave=False)):
    #     pbar_year.set_description(f" year")

    #     year_df = df.loc[year_mask[year]]

    #     # split yearly only, don't need to proceed further
    #     if not split_monthly:
    #         year_out_stem = f'{out_head}_{year}_{out_tail}'
    #         year_df.to_parquet(out_dir_parquet / f'{year_out_stem}.parquet')
    #         year_df.to_csv(out_dir_csv / f'{year_out_stem}.csv', index=False)
    #         del year_df
    #         continue

        # split monthly, need to proceed further
        # year_out_dir_csv = out_dir_csv / f'{year:04d}'
        # year_out_dir_parquet = out_dir_parquet / f'{year:04d}'
        # year_out_dir_csv.mkdir(parents=True, exist_ok=True)
        # year_out_dir_parquet.mkdir(parents=True, exist_ok=True)
            
        # for month in (pbar_month := tqdm(range(1, 13), leave=False)):
        #     pbar_month.set_description(f"month")

        #     month_df = year_df.loc[month_mask[month]]

        #     # split monthly only, don't need to proceed further
        #     if not split_daily:
        #         month_out_stem = f'{out_head}_{year}_{month:02d}_{out_tail}'
        #         month_df.to_parquet(year_out_dir_parquet / f'{month_out_stem}.parquet')
        #         month_df.to_csv(year_out_dir_csv / f'{month_out_stem}.csv', index=False)
        #         del month_df
        #         continue

        #     # split daily, need to proceed further

    month_out_dir_csv = out_dir_csv / str(year) / f'{month:02d}'
    month_out_dir_parquet = out_dir_parquet / str(year) / f'{month:02d}'
    month_out_dir_csv.mkdir(parents=True, exist_ok=True)
    month_out_dir_parquet.mkdir(parents=True, exist_ok=True)

    _, num_days = monthrange(year, month)

    for day in (pbar_day := tqdm(range(1, num_days+1), leave=False)):
        pbar_day.set_description(f"  day")

        day_df = df.filter(day_mask[day])
        
        day_out_stem = f'{out_head}_{year}_{month:02d}_{day:02d}_{out_tail}'
        day_df.write_parquet(month_out_dir_parquet / f'{day_out_stem}.parquet')
        day_df.write_csv(month_out_dir_csv / f'{day_out_stem}.csv')

        del day_df

        #     del month_df

        # del year_df



split_by_day = functools.partial(split_by_time_unified, time_res='daily')
# split_by_month = functools.partial(split_by_time_unified, time_res='monthly')
# split_by_year = functools.partial(split_by_time_unified, time_res='yearly')

# NOTE run first after OT data has been downloaded
if __name__ == "__main__":
    out_dir = "../data/bts_raw/airport_split_by_day"

    # df = pl.read_csv(testing_path)
    for year in tqdm(range(2019, 2020), desc="Processing years"):
        for month in tqdm(range(1, 13), desc="Processing months"):
            testing_path = f"../data/bts_raw/On_Time_Reporting_Carrier_On_Time_Performance_(1987_present)_{str(year)}_{str(month)}.csv"

            for code in tqdm(list(CoreAirports.keys()), desc="Processing airports"):
                df, save_path_parquet, save_path_csv = filter_for_airport(testing_path, code, out_dir)
                save_path = str(save_path_parquet) + '.parquet'
                split_by_day(save_path, out_dir=out_dir)

# %%
