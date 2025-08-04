# %%
from pathlib import Path
import pandas as pd
from bayes_air.schedule import split_and_parse_full_schedule

if __name__ == "__main__":
    current_file_path =  Path(__file__).resolve()
    data_dir = current_file_path.parent.parent / "data"
    # 7/16-23/19 seems to have a range of delays
    parquet_dir = data_dir / "bts_remapped/lga_reduced_2010-2019_clean_daily/parquet" 
    schedule_path = parquet_dir / "2019/07/lga_reduced_2019_07_23_clean.parquet"

    print(f"Attempting to read from: {schedule_path}")
    print(f"Does the path exist? {schedule_path.exists()}")


    schedule_df = pd.read_parquet(schedule_path)

    (
        network_flights, network_airports,
        incoming_flights, source_supernode,
    ) = \
        split_and_parse_full_schedule(
            schedule_df, ["LGA"],
        )
    
    print(len(network_flights))
    print([na.code for na in network_airports])
    print(len(incoming_flights))
    print(source_supernode.dest_codes)