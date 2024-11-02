import os
import pandas as pd
import numpy as np
from scipy.signal import find_peaks

def sort_file(directory):

    # list all files in the given directory 
    files = os.listdir(directory) 

    # filter for CSV files
    phases_files = [f for f in files if f.endswith('.csv')]
    
    # sort files based on the first number in the filename
    sorted_files = sorted(phases_files, key=lambda x: int(x.split('_')[1]))
    
    return sorted_files

def data_concat(file_list):
    
    # relative Root Directory Path
    rrpath = os.path.join('..')
    
    # function to read and process each CSV file
    def read_and_process(file_path):
        df = pd.read_csv(file_path, header=None)
        df.columns = ['datetime', 'optical distance along cable (m)']
        df = df.drop_duplicates(subset=['datetime'])
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values(by='datetime').reset_index(drop=True)
        df.set_index('datetime', inplace=True)
        return df

    # list to hold processed DataFrames
    processed_dfs = []

    # process each file in the list
    for file_name in file_list:
        file_path = os.path.join(rrpath, 'data', 'phase_time_series_lower',file_name) ## change folder if needed
        processed_df = read_and_process(file_path)
        processed_dfs.append(processed_df)

    # concatenate all processed DataFrames
    concatenated_df = pd.concat(processed_dfs)
    
    # sort the concatenated DataFrame by datetime
    concatenated_df = concatenated_df.sort_index()

    return concatenated_df

def prep_data(concatenated_df, window='60s'):
    resample_df = concatenated_df.resample(window).mean()
    interpolate_df = resample_df.interpolate()

    return interpolate_df
 