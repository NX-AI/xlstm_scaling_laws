import pickle
from typing import Literal

import pandas as pd

from .datafiles import data_dir

inference_time_data_file = data_dir / "inference_results_df_dict.pkl"

def create_inference_data_dfs(raw_data_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:

    ttft_df = raw_data_df.copy()

    # assign new variable runtime with the same values as the "ttft" column
    ttft_df[('measured_data','runtime')] = ttft_df[('measured_data','ttft')]

    # drop "steptime" and "throughput" columns
    ttft_df.drop(columns=[('measured_data','ttft'), ('measured_data', 'steptime'), ('measured_data', 'throughput')], inplace=True)
    
    ttft_df.loc[ttft_df[('measured_data', 'runtime')] <= 0, ('measured_data', 'runtime')] = float('nan')  # Set negative or zero step times to NaN


    avg_step_time_df = raw_data_df.copy()

    # assign new variable runtime with the same values as the "steptime" column
    avg_step_time_df[('measured_data','runtime')] = avg_step_time_df[('measured_data','steptime')]
    # drop "ttft" and "throughput" columns
    avg_step_time_df.drop(columns=[('measured_data', 'ttft'), ('measured_data', 'steptime'), ('measured_data', 'throughput')], inplace=True)

    avg_step_time_df.loc[avg_step_time_df[('measured_data', 'runtime')] <= 0, ('measured_data', 'runtime')] = float('nan')  # Set negative or zero step times to NaN

    
    return ttft_df, avg_step_time_df


def load_inference_time_data(model_type: Literal["xlstm", "llama2"]) -> tuple[pd.DataFrame, pd.DataFrame]:
    with open(inference_time_data_file, "rb") as f:
        inference_data_df_dict = pickle.load(f)

    return create_inference_data_dfs(inference_data_df_dict[model_type])