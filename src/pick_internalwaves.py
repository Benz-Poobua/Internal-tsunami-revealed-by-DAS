# import packages
import h5py, scipy, datetime, csv, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
sys.path.append("../utils/")
import DASfuncs

# get files
t_start = datetime.datetime(2023,8,10)
t_end = datetime.datetime(2023,8,11)
path = '../data/DAS/data_1Hz'
files = DASfuncs.get_Onyx_h5(path, t_start, t_end)
print('{} files in directory'.format(len(files)))

# read the data
time_read, data_read, attrs_read = DASfuncs.read_Onyx_h5_to_list(files, cha_start=None, cha_end=None, t_start=t_start, t_end=t_end, verbose=True)
# concatenate files
t_rec, data_rec, attrs = DASfuncs.comb_Onyx_data(time_read, data_read, attrs_read)

# fill contiuous parts of data into array
time_list, data_list = DASfuncs.split_continuous_data(t_rec, data_rec, attrs)
# fill data gaps in array
times_filled, data_filled = DASfuncs.fill_data_gaps(time_list, data_list, attrs, t_format='datetime')

# filtering
sos = scipy.signal.butter(2, 0.1,'lowpass', fs=attrs['PulseRate'], output='sos')
filt_list = [DASfuncs.apply_sosfiltfilt_with_nan(sos, arr, axis=0) for arr in data_list]

times_filled, data_filled = DASfuncs.fill_data_gaps(time_list, filt_list, attrs, t_format='datetime')

data_arr = data_filled
times = DASfuncs.sintela_to_datetime(t_rec)  # Convert timestamp to datetime
dx = attrs['SpatialSamplingInterval']
chas = np.arange(attrs['StartLocusIndex'], attrs['StartLocusIndex']+attrs['NumberOfLoci'])
dists = chas*dx

# Slice data in time and space
start_dist = 2820
end_dist = 3150

start_time = t_start
end_time = t_end

t_idx_start = np.argmin(np.abs(times-start_time))
t_idx_end = np.argmin(np.abs(times-end_time))
d_idx_start = np.argmin(np.abs(dists-start_dist))
d_idx_end = np.argmin(np.abs(dists-end_dist))

plot_arr = data_arr[t_idx_start:t_idx_end, d_idx_start:d_idx_end]
plot_times = times[t_idx_start:t_idx_end]
plot_dists = dists[d_idx_start:d_idx_end]

data_norm = plot_arr #- np.nanmedian(plot_arr, axis=0)
# data_norm = data_norm / np.std(data_norm, axis=0)[None,:]

import csv

# Initialize an empty list to store cursor positions
positions = []

# Define the event handler for mouse clicks
def onclick(event):
    # Get the x and y coordinates of the click
    x, y = event.xdata, event.ydata
    if x is not None and y is not None:  # Check if the click is within the plot area
        # Convert x from float to datetime
        dt = mdates.num2date(x).strftime('%Y-%m-%d %H:%M:%S')
        positions.append([dt, y])
        print(f'Position saved: datetime={dt}, y={y}')
        
        # Save the positions to a CSV file
        with open('../data/phase_time_series/positions.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerows([[dt, y]])  # Write the new data

# Plot
fig, ax = plt.subplots(figsize=(2.5*6.4,1.5*4.8))

im = ax.imshow(data_norm.T, aspect='auto',
             origin='lower',
             vmin=np.percentile(data_norm[~np.isnan(data_norm)],5),
             vmax=np.percentile(data_norm[~np.isnan(data_norm)],95),
             extent=[plot_times[0], plot_times[-1],
                  plot_dists[0], plot_dists[-1]],
             cmap='RdBu_r',
             # interpolation='none',
               zorder=0
             )

ax.set_ylabel('Optical Distance along Cable [m]')
ax.set_xlabel('Time')
ax.set_title('Event after  {}'.format(plot_times[0].strftime("%Y/%m/%dT%H:%M:%S")))

cbar = fig.colorbar(im, pad=0.01)
cbar.set_label('Phase [rad]')

# Connect the event handler to the figure
cid = fig.canvas.mpl_connect('button_press_event', onclick)

# Show the plot
plt.show()
