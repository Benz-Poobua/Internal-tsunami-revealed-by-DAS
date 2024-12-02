import os, sys, datetime, glob, h5py, obspy, re
import scipy.interpolate
import scipy.signal
import numpy as np

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