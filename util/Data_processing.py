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
        df.columns = ['datetime', 'phase (rad)']
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
    resample_df = concatenated_df.resample('60s').mean()
    interpolate_df = resample_df.interpolate()

    return interpolate_df

def compute_fft(phases_df):
    """
    Compute the FFT of the phase values in the given DataFrame.
    
    Parameters:
    phases_df (DataFrame): DataFrame containing 'datetime' and 'phase (rad)' columns.
    
    Returns:
    positive_frequencies (ndarray): Array of positive frequency values.
    positive_magnitude (ndarray): Array of corresponding magnitudes.
    """
    # extract phase values and time index
    phase_values = phases_df['phase (rad)'].values
    time_index = phases_df.index
    
    # calculate time difference (seconds)
    dt = (time_index[1] - time_index[0]).total_seconds()  # time difference in seconds; uniformly sampled
    n = len(phase_values)  # number of samples
    
    # perform the FFT
    fft_values = np.fft.fft(phase_values)
    fft_magnitude = np.abs(fft_values) / n  # normalize the amplitude
    #fft_magnitude /= np.max(fft_magnitude) # optional
    
    # get frequency bins
    fft_frequency = np.fft.fftfreq(n, d=dt)  # frequency bins
    
    # only keep the positive values
    positive_frequencies = fft_frequency[:n // 2]
    positive_magnitude = fft_magnitude[:n // 2]
    
    return positive_frequencies, positive_magnitude

def peak_iden(positive_frequencies):
    
    f_min = 1/(60*60)
    f_max = 1/(5*50)

    freq_mask = (positive_frequencies >= f_min) & (positive_frequencies <= f_max)

    peaks, _ = find_peaks(freq_mask, threshold=np.log10(3))

    top_peaks = peaks[:3] 

    peak_values = positive_frequencies[top_peaks]

    print("Indices of local maxima:", top_peaks)
    print("Values of local maxima:", peak_values)

    return top_peaks, peak_values
