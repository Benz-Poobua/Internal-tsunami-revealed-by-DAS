import numpy as np
import scipy.signal
import scipy.interpolate
import datetime
import os
import sys
import glob
import h5py
import obspy
import re

# def sintela_to_datetime(sintela_times):
#     '''returns an array of datetime.datetime ''' 
#     days1970 = datetime.date(1970, 1, 1).toordinal()
#     # Vectorize everything
#     converttime = np.vectorize(datetime.datetime.fromordinal)
#     addday_lambda = lambda x : datetime.timedelta(days=x)
#     adddays = np.vectorize(addday_lambda)
#     day = days1970 + sintela_times/1e6/60/60/24
#     thisDateTime = converttime(np.floor(day).astype(int))
#     dayFraction = day-np.floor(day)
#     thisDateTime = thisDateTime + adddays(dayFraction)
#     return thisDateTime

def sintela_to_datetime(sintela_times):
    '''returns a datetime.datetime object if input is a Numpy integer,
       or an array of datetime.datetime objects if input is a NumPy array'''

    # Check if input is a single integer
    if isinstance(sintela_times, np.integer):
        return datetime.datetime.utcfromtimestamp(sintela_times / 1e6)
    # Check if input is a NumPy array
    elif isinstance(sintela_times, np.ndarray):
        # Apply datetime.datetime.utcfromtimestamp() to all entries of the array
        datetime_arr = np.vectorize(datetime.datetime.utcfromtimestamp)(sintela_times / 1e6)
        return datetime_arr
    else:
        raise ValueError("Input must be either a single NumPy integer or a NumPy array")

def get_Onyx_file_time(file):
    '''Find timestamp in Onyx file name.'''
    # Define regular expression pattern to match date and time components
    pattern = r'.*(\d{4})\D+(\d{1,2})\D+(\d{1,2})\D+(\d{1,2})\D+(\d{1,2})\D+(\d{1,2}).*'
    filename = os.path.basename(file)
    # Extract date and time components using regular expression
    match = re.match(pattern, filename)
    if match:
        # Extract matched groups
        groups = match.groups()
        # Convert matched groups to integers
        year, month, day, hour, minute, second = map(int, groups)
        # Create datetime object
        datetime_obj = datetime.datetime(year, month, day, hour, minute, second)
        return datetime_obj
    else:
        print("Invalid datetime format")
        return None
    
# def get_Onyx_file_time(filepath):
#     '''returns datetime.datetime of filename'''
#     filename = os.path.basename(filepath)
#     date = filename.split('_')[1]
#     time = filename.split('_')[2]
#     time = datetime.datetime.strptime('{}T{}'.format(date, time), '%Y-%m-%dT%H.%M.%S')
#     return time

# def get_Onyx_h5(dir_path, t_start, t_end=None, length=60, filelength=60):
#     '''get list of availlable files within time range'''
#     if not t_end:
#         t_end = t_start + datetime.timedelta(seconds=length)
#     files = []
#     for file in glob.iglob(os.path.join(dir_path,'*.h5'), recursive=True):
#         file_timestamp = get_Onyx_file_time(file)
#         if (file_timestamp>=t_start-datetime.timedelta(seconds=filelength)) & \
#             (file_timestamp<=t_end):
#             files.append(file)
#         else:
#             pass
#     files.sort()
#     # if len(files)=0:
#     #     print('No files in list. Maybe the filelength was set incorrectly. Default is 60 seconds.')
#     return files

def get_Onyx_h5(dir_path, t_start, t_end=None, length=60):
    '''get list of availlable files within time range'''
    if not t_end:
        t_end = t_start + datetime.timedelta(seconds=length)
    all_files = glob.glob(os.path.join(dir_path,'*.h5'), recursive=True)
    out_files = []
    for i,file in enumerate(all_files):
        file_timestamp = get_Onyx_file_time(file)
        if t_start <= file_timestamp <=t_end:
            # Add previous file to first valid file
            if not out_files and i > 0:
                out_files.append(all_files[i-1])
            out_files.append(file)
        else:
            pass
    out_files.sort()
    return out_files


# read multiple files
def read_Onyx_h5_to_list(files, cha_start=None, cha_end=None, t_start=None, t_end=None, verbose=False):
    '''reads lists of hdf5 files saved by the Sintela Onyx DAS interrogator, outputs lists of data'''
    data_read = []
    time_read = []
    attrs_read = []
    for i,file in enumerate(files):
        if verbose:
            print("File {} of {}".format(i+1, len(files)), end="\r")
        try:
            f = h5py.File(file,'r')
            # check data quality
            if not f['Acquisition/Raw[0]/RawDataTime'].shape[0] == f['Acquisition/Raw[0]/RawData'].shape[0]:
                f.close()
            else:
                # slice time if necessary
                time_rec = np.array(f['Acquisition/Raw[0]/RawDataTime'], dtype='int64')
                t_datetime = sintela_to_datetime(time_rec)
                if t_start:
                    t_start_idx = np.min(np.where(np.array([(t-t_start).total_seconds() for t in t_datetime])>0))
                else: t_start_idx = 0
                if t_end:
                    t_end_idx = np.max(np.where(np.array([(t-t_end).total_seconds() for t in t_datetime])<0))+1
                else:
                    t_end_idx = None

                data_rec = np.array(f['Acquisition/Raw[0]/RawData'], dtype='float32')[t_start_idx: t_end_idx, cha_start:cha_end]
                time_rec = time_rec[t_start_idx: t_end_idx]
                attrs_rec = dict(f['Acquisition'].attrs)
                f.close()
                if cha_start:
                    attrs_rec['StartLocusIndex'] = cha_start
                    attrs_rec['NumberOfLoci'] = data_rec.shape[1]

                if len(time_rec) > 0:
                    time_read.append(time_rec)
                    data_read.append(data_rec)
                    attrs_read.append(attrs_rec)
                # del time_rec, data_rec, attrs_rec, t_datetime # delete the variables used to read data
        except Exception as e:
            if verbose:
                print('Problems with: {}'.format(file))
                print(e)
            continue
    return time_read, data_read, attrs_read

def pad_array_with_nans(arr, target_shape):
    if arr.shape[0] > target_shape[0] or arr.shape[1] > target_shape[1]:
        raise ValueError("Target shape should be larger than the original array.")
    # Create a new array filled with NaNs
    padded_arr = np.full(target_shape, np.nan)
    # Copy the values from the original array to the padded array
    padded_arr[:arr.shape[0], :arr.shape[1]] = arr
    return padded_arr

def comb_Onyx_data(time_read, data_read, attrs_read):
    '''combine data from files with identical attributes'''
    compare_attrs = [(attrs['PulseRate'], np.round(attrs['SpatialSamplingInterval'],2), 
                            attrs['StartLocusIndex']) for attrs in attrs_read]
    if all(i == compare_attrs[0] for i in compare_attrs):
        maxcha = np.max([arr.shape[1] for arr in data_read])
        times = np.concatenate(time_read, axis=0)
        data = np.concatenate([pad_array_with_nans(arr, (arr.shape[0], maxcha)) for arr in data_read], axis=0)
        attrs = attrs_read[0]
    else:
        return None
    return times, data, attrs

def split_continuous_data(t_rec, data_rec, attrs):
    '''fills continuous (without data gaps) data into list'''
    data_list = []
    time_list = []

    # find indices where time difference is larger than PulseRate and split there
    dt = np.diff(t_rec)/1e6
    gap_idx = np.where(np.abs(dt)>1/attrs['PulseRate'])[0]
    
    if len(gap_idx)>0: # if data gaps present
        time_list.append(t_rec[:gap_idx[0]]) # first gapless data piece
        data_list.append(data_rec[:gap_idx[0]]) 
        for i in range(0,len(gap_idx)-1):
            time_list.append(t_rec[gap_idx[i]+1:gap_idx[i+1]+1]) # gapless data in between, +1 because of np.diff index shifted
            data_list.append(data_rec[gap_idx[i]+1:gap_idx[i+1]+1]) 
        data_list.append(data_rec[gap_idx[-1]+1:]) # last gapless data piece
        time_list.append(t_rec[gap_idx[-1]+1:])

        time_list = [i for i in time_list if len(i)>0] # remove empty list entries
        data_list = [i for i in data_list if len(i)>0] 
    else: # if no data gaps present
        data_list = [data_rec]
        time_list = [t_rec]
    return time_list, data_list

def fill_data_gaps(time_list, data_list, attrs, fill_value=np.nan, t_format=None):
    '''convert lists of gapless arrays into one array with fill_value at gaps'''
    # if len(time_list)>1:
    dt_eq = 1/attrs['PulseRate']*1e6
    t_eq = np.arange(time_list[0][0], time_list[-1][-1]+dt_eq, dt_eq) # equally sampled time array

    arr_filled = np.full((len(t_eq),data_list[0].shape[1]), np.nan) # array where to write the data to
    i=0 # index of arr_new
    for t_arr, d_arr in zip(time_list, data_list):
        while t_arr[0] > t_eq[i]: # fill with data only if recorded time is in equally sampled array (to within accuracy)
            i+=1
        minidx = np.min(np.where(t_arr>=t_eq[i])) # next two lines, because some timestamps are double
        lenidx = len(np.where(t_arr>=t_eq[i])[0])
        arr_filled[i:i+lenidx] = d_arr[minidx:]
    #     arr_filled[i:i+len(t_arr)] = d_arr # old version
    #     i+=len(t_arr)
        i+=lenidx

    if t_format=='datetime':
        t_eq = sintela_to_datetime(t_eq)
    else:
        pass
        
    return t_eq, arr_filled

def get_gaps(time_list, attrs, t_format=None):
    '''creates list with start and end times of data gaps'''
    gap_list = []
    dt_eq = 1/attrs['PulseRate']*1e6
    for i in range(len(time_list)-1):
        gap_list.append((time_list[i][-1]+dt_eq, time_list[i+1][0]-dt_eq)) # times of gaps
    if t_format=='datetime':
        gap_list = [(sintela_to_datetime(s), sintela_to_datetime(e)) for (s,e) in gap_list]
    return gap_list

def apply_sosfiltfilt_with_nan(sos, data, axis=-1, padtype='odd', padlen=None, verbose=True):
    '''applies the scipy.sosfiltfilt function and ignores errors'''
    try:
        filtered_data = scipy.signal.sosfiltfilt(sos, data, axis=axis, padtype=padtype, padlen=padlen)
        return filtered_data
    except Exception as e:
        if verbose:
            warning_str = 'Returning an array filled with np.nan equally sized to the input array.'
            print('{}, {}'.format(e, warning_str), end='\r')
        return np.full(data.shape, np.nan)
    
def decimate(time_list, data_list, factor, attrs):
    '''decimates list of gapless arrays in time by factor and outputs one array with filled gaps'''
    sos = scipy.signal.butter(2, attrs['PulseRate']/factor/2.,'lowpass', fs=attrs['PulseRate'], output='sos') # frequency in m
    filt_list = [apply_sosfiltfilt_with_nan(sos, arr, axis=0) for arr in data_list]
    # fill filtered data into array
    t_cont, data_filt = fill_data_gaps(time_list, filt_list, attrs, t_format='datetime')
    data_dec = data_filt[::factor,:]
    t_dec = t_cont[::factor]
    return t_dec, data_dec
    
def interp_gaps(times_filled, data_filled, max_gap, **kwargs):
    '''interpolates NaNs in data up to a maximum data gap length in samples'''
    # select data gaps with a certain length
    gap_idxs = np.where(np.isnan(data_filled).all(axis=1))[0] # find all NaNs
    gaps_selected = []
    gap = []
    for i in range(len(gap_idxs)-1): # keep gap if shorter than max_gap
        gap.append(gap_idxs[i])
        if (gap_idxs[i+1]-gap_idxs[i] > 1):
            if len(gap)<max_gap:
                gaps_selected.extend(gap)
            gap = []

    if len(gaps_selected) > 0:
        # create boolean array from data gap indices
        gaps_bool = np.full(len(times_filled), False)
        gaps_bool[gaps_selected] = True

        # interpolate selected data gaps
        f_interp = scipy.interpolate.interp1d(times_filled[~gaps_bool], data_filled[~gaps_bool,:], axis=0)
        data_interp = data_filled.copy()
        data_interp[gaps_bool,:] = f_interp(times_filled[gaps_bool])
        return data_interp
    else:
        return data_filled

def split_at_datagaps(times_filled, data_filled):
    '''splits arrays with NaNs into a list of contiguous arrays'''
    mask1d = np.isnan(data_filled).all(axis=1)
    new_data_list = [data_filled[s,:] for s in np.ma.clump_unmasked(np.ma.array(times_filled, mask=mask1d))]
    new_time_list = [times_filled[s] for s in np.ma.clump_unmasked(np.ma.array(times_filled, mask=mask1d))]
    return new_time_list, new_data_list

def digit_count(number):
    '''Counts number of digits of an integer'''
    if number == 0:  # Special case for handling zero
        return 1
    count = 0
    number = abs(number)  # Convert the number to its absolute value
    while number > 0:
        count += 1
        number //= 10
    return count

def format_with_leading_zeros(number, leading_zeros):
    '''Formats a number with leading zeros'''
    num_str = str(number)
    num_digits = len(num_str)
    
    if leading_zeros <= num_digits:
        return num_str
    
    num_zeros = leading_zeros - num_digits
    formatted_str = '0' * num_zeros + num_str
    return formatted_str

# This function has a problem. Stats.channel can only have 3 digits.
def h5toStream(time_list, data_list, attrs, network='XH', channel=''):
    '''Converts a DAS data array file into an obspy stream'''
    st = obspy.Stream()
    for times, data in zip(time_list, data_list): # loop over continuous (in time) data pieces
        for cha_idx in range(data.shape[1]): # loop over DAS channels
            tr = obspy.Trace()
            # set the stats
            stats = obspy.core.trace.Stats()
            stats.network = network
            stats.station = format_with_leading_zeros(cha_idx, digit_count(data.shape[1]))
            stats.location = ''
            stats.channel = channel
            stats.sampling_rate = attrs['PulseRate']
            stats.npts = len(times)
            stats.starttime = obspy.UTCDateTime(sintela_to_datetime(times[0]))
            stats.gauge_length = attrs['GaugeLength']
            stats.spatial_sampling_interval = attrs['SpatialSamplingInterval']
            # add data
            tr.data = data[:,cha_idx]
            tr.stats = stats
            # append trace to stream
            st.append(tr)
    st.merge()
    st.sort()
    return st

def channel_number(files):
    '''get the number of DAS channels for files in list'''
    n_chas = np.full(len(files), np.nan)
    for i,file in enumerate(files):
        print("File {} of {} processed for reading channel count.".format(i+1, len(files)), end="\r")
        try:
            f = h5py.File(file,'r')
            n_cha = dict(f['Acquisition'].attrs)['NumberOfLoci']
            n_chas[i] = n_cha
            f.close()
        except Exception as e:
            print(e)
            print(i, file)
            continue
    return n_chas

def calc_std(file):
    '''filter above 500Hz and calculate standard deviation file-wise'''
    try:
        # read the data (really simple and stupid)
        f = h5py.File(file,'r')
        time_read = np.array(f['Acquisition/Raw[0]/RawDataTime'], dtype='int64')
        data_read = np.array(f['Acquisition/Raw[0]/RawData'], dtype='float32')   
        attrs_read = dict(f['Acquisition'].attrs)
        f.close()
        # fill contiuous parts of data into separate arrays
        time_list, data_list = split_continuous_data(time_read, data_read, attrs_read)
        # filtering
        sos = scipy.signal.butter(2, 500, 'highpass', fs=attrs_read['PulseRate'], output='sos')
        filt_list = [apply_sosfiltfilt_with_nan(sos, arr, axis=0) for arr in data_list]
        # simple concatenation of data into one array
        data_arr = np.concatenate(filt_list, axis=0)
        # calculate absolute median
        std = np.nanstd(data_arr, axis=0)
        t = time_read[0]
    except Exception as e:
        print('Problems with: {}'.format(file))
        print(e)
        return None
    return t, std

def fk(data, dt, dx, taper=0.05):
    """
    Compute and plot the f-k transform of a 2D space-time data array.

    Input:
    ------
    d: data matrix organised as d[space indices, time indices]
    dt: time increment [s]
    dx: space increment [m]

    Output:
    -------
    f, k: frequency and wave number arrays
    ft: f-k transformed data
    """
    d = data.copy()
    
    nt=np.shape(d)[0]
    nx=np.shape(d)[1]

    # Apply a spatial taper to the data.
    width=int(nx*taper)
    for i in range(width): 
        d[:,i]=((i+1)/(width+1))*d[:,i]
        d[:,nx-i-1]=((i+1)/(width+1))*d[:,nx-i-1]

    # Apply a temporal taper to the data.
    width=int(nt*taper)
    for i in range(width): 
        d[i,:]=((i+1)/(width+1))*d[i,:]
        d[nt-i-1,:]=((i+1)/(width+1))*d[nt-i-1,:]
        
    ft = np.fft.fftshift(np.fft.fft2(d))
    f = np.fft.fftshift(np.fft.fftfreq(d.shape[0], d=dt))
    k = np.fft.fftshift(np.fft.fftfreq(d.shape[1], d=dx))
    
    return f,k,ft




##### RECYCLING ######

# this might be deleted    
def _fill_data_gaps_with_nans(f):
    # fill data gaps with NaN values
    data = np.array(f['Acquisition/Raw[0]/RawData']) # read this in again to avoid malefunction if cell is run again
    t_rec = np.array(f['Acquisition/Raw[0]/RawDataTime'])
    dt_eq = 1/attrs['PulseRate']*1e6

    t_eq = np.arange(t_rec[0], t_rec[-1]+dt_eq, dt_eq) # equally sampled time array
    t_new = np.full(len(t_eq), np.nan) # same length array filled with NaN
    arr_new = np.full((len(t_eq),data.shape[1]), np.nan) # array where to write the data to

    i=0 # index of arr_new and t_new
    for t_rec_idx in range(len(t_rec)):   
        while t_rec[t_rec_idx] > t_eq[i]: # fill with data only if recorded time is in equally sampled array (to within accuracy)
            i+=1
        t_new[i] = t_rec[t_rec_idx]
        arr_new[i] = data[t_rec_idx,:]
    data = arr_new # override the data array
    times = sintela_to_datetime(t_eq)
    return

def _get_Onyx_file_time(filepath):
    '''returns datetime.datetime of filename'''
    filename = os.path.basename(filepath)
    date = filename.split('_')[1]
    time = filename.split('_')[2]
    time = datetime.datetime.strptime('{}T{}'.format(date, time), '%Y-%m-%dT%H.%M.%S')
    return time