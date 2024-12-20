# Import necessary dependencies
import h5py, scipy, datetime, csv, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
sys.path.append("../utils/")
import DASfuncs

# Set paths and date range for DAS data
t_start = datetime.datetime(2023,8,17) # Start time for DAS data
t_end = datetime.datetime(2023,8,18) # End time for DAS data
path = '../data/DAS/data_1Hz'
position_filename = f'../data/phase_time_series_lower/positions_{t_start.day}_{t_end.day}.csv'
files = DASfuncs.get_Onyx_h5(path, t_start, t_end)
print('{} files in directory'.format(len(files)))

# Load DAS data
time_read, data_read, attrs_read = DASfuncs.read_Onyx_h5_to_list(files, cha_start=None, cha_end=None, t_start=t_start, t_end=t_end, verbose=True)
t_rec, data_rec, attrs = DASfuncs.comb_Onyx_data(time_read, data_read, attrs_read)
time_list, data_list = DASfuncs.split_continuous_data(t_rec, data_rec, attrs)
times_filled, data_filled = DASfuncs.fill_data_gaps(time_list, data_list, attrs, t_format='datetime')
sos = scipy.signal.butter(2, 0.1,'lowpass', fs=attrs['PulseRate'], output='sos')
filt_list = [DASfuncs.apply_sosfiltfilt_with_nan(sos, arr, axis=0) for arr in data_list]
times_filled, data_filled = DASfuncs.fill_data_gaps(time_list, filt_list, attrs, t_format='datetime')
data_arr = data_filled
times = DASfuncs.sintela_to_datetime(t_rec)
dx = attrs['SpatialSamplingInterval'] # Gauge Length
chas = np.arange(attrs['StartLocusIndex'], attrs['StartLocusIndex']+attrs['NumberOfLoci'])
dists = chas*dx

# Select DAS slice
start_dist, end_dist = 2820, 3150 # Distance range to plot
t_idx_start, t_idx_end = np.argmin(np.abs(times-t_start)), np.argmin(np.abs(times-t_end))
d_idx_start, d_idx_end = np.argmin(np.abs(dists-start_dist)), np.argmin(np.abs(dists-end_dist))
plot_arr_DAS = data_arr[t_idx_start:t_idx_end, d_idx_start:d_idx_end]  # DAS data to plot
plot_times_DAS = times[t_idx_start:t_idx_end] # Time values for DAS plot
plot_dists_DAS = dists[d_idx_start:d_idx_end] # Distance values for DAS plot
data_norm_DAS = plot_arr_DAS

# Load DTS data
infile = '../data/DTS/temp_cal_valid_cable_rmnoise.csv'
df_temp = pd.read_csv(infile, index_col=0)
df_temp.columns = pd.to_datetime(df_temp.columns)  # Convert column names (time) to datetime
temp_arr = df_temp.to_numpy().T  # Convert DataFrame to NumPy array
times_DTS = pd.to_datetime(df_temp.columns) # Time values for DTS
dists_DTS = df_temp.index.to_numpy() # Distance values for DTS
t_idx_start_DTS, t_idx_end_DTS = np.argmin(np.abs(times_DTS-t_start)), np.argmin(np.abs(times_DTS-t_end))
d_idx_start_DTS, d_idx_end_DTS = np.argmin(np.abs(dists_DTS-start_dist)), np.argmin(np.abs(dists_DTS-end_dist))
plot_arr_DTS = temp_arr[t_idx_start_DTS:t_idx_end_DTS, d_idx_start_DTS:d_idx_end_DTS] # DTS data to plot
plot_times_DTS = times_DTS[t_idx_start_DTS:t_idx_end_DTS] # Time values for DTS plot
plot_dists_DTS = dists_DTS[d_idx_start_DTS:d_idx_end_DTS] # Distance values for DTS plot

# Initialize plotting
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# Plot DAS
im_DAS = ax1.imshow(data_norm_DAS.T, aspect='auto', origin='lower', vmin=np.percentile(data_norm_DAS[~np.isnan(data_norm_DAS)], 5), vmax=np.percentile(data_norm_DAS[~np.isnan(data_norm_DAS)], 95), extent=[plot_times_DAS[0], plot_times_DAS[-1], plot_dists_DAS[0], plot_dists_DAS[-1]], cmap='RdBu_r')
ax1.set_title('DAS Data')
ax1.set_ylabel('Optical Distance along Cable [m]')
cbar_DAS = fig.colorbar(im_DAS, ax=ax1, pad=0.01) # Colorbar for DAS plot
cbar_DAS.set_label('Phase [rad]')
ax1.invert_yaxis() # Invert y-axis to match the cable orientation

# Plot DTS
im_DTS = ax2.imshow(plot_arr_DTS.T, aspect='auto', origin='lower', vmin=np.percentile(plot_arr_DTS[~np.isnan(plot_arr_DTS)],1), vmax=np.percentile(plot_arr_DTS[~np.isnan(plot_arr_DTS)],99), extent=[plot_times_DTS[0], plot_times_DTS[-1], plot_dists_DTS[0], plot_dists_DTS[-1]], cmap='plasma')
ax2.set_title('DTS Data')
ax2.set_ylabel('Optical Distance along Cable [m]')
ax2.set_xlabel('Time')
cbar_DTS = fig.colorbar(im_DTS, ax=ax2, pad=0.01)
cbar_DTS.set_label('Temperature [°C]')
ax2.invert_yaxis() # Invert y-axis to match the cable orientation

# Initialize positions list and hold flag
on_hold = False
positions = []

# Click handler for data collection
def onclick(event):
    """
    Handles mouse click events to collect data points (phase and datetime).
    Saves the data point in a CSV file and adds a red dot on both DAS and DTS plots.
    
    param:
    - event: The mouse click event
    """
    if event.inaxes == ax1 and not on_hold:
        x, y = event.xdata, event.ydata
        if x is not None and y is not None:
            dt = mdates.num2date(x).strftime('%Y-%m-%d %H:%M:%S') # Convert x-axis (time) to datetime
            positions.append([dt, y]) # Store datetime and distance
            print(f'Position saved: datetime={dt}, y={y}')
            with open(position_filename, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerows([[dt, y]])

        # Add a red dot at the clicked position on both DAS and DTS plots
        ax1.plot(x, y, 'r.')
        ax2.plot(x, y, 'r.')
        plt.draw()

# Cursor movement handler to synchronize cursor across plots
def on_motion(event):
    """
    Synchronizes the cursor movement between DAS and DTS plots. 
    Displays a vertical line on the DTS plot at the same x position as the cursor in the DAS plot.
    
    param:
    - event: The mouse motion event
    """
    if event.inaxes == ax1:
        ax2.axvline(event.xdata, color='gray', linestyle='--', lw=0.5)  # Draw a vertical line on DTS plot
        plt.draw()

def on_hold_toggle(event):
    
    """
    Toggles the hold state for data collection. If on hold, data collection is paused.
    
    param:
    - event: The key press event to toggle hold
    """

    global on_hold
    on_hold = not on_hold
    print(f"Hold state: {'ON' if on_hold else 'OFF'}")

# Connect events
cid_click = fig.canvas.mpl_connect('button_press_event', onclick) # Mouse click event
cid_motion = fig.canvas.mpl_connect('motion_notify_event', on_motion) # Mouse motion event
cid_hold = fig.canvas.mpl_connect('key_press_event', on_hold_toggle) # Key press event to toggle hold state

plt.show()
