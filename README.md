# Internal-tsunami-revealed-by-DAS
## 1. Introduction

### Inspiration
DAS data captures phase changes resulting from the Rayleigh backscatter of pulses transmitted along the cable. These pulses are influenced by environmental factors, leading to changes in phase or strain along the cable. On the other hand, DTS data measures temperature variations around the cable, which, in turn, influence the density of the surrounding ocean water. Temperature gradients detected by the DTS system can help delineate the vertical structure of the ocean, with distinct temperature layers forming interfaces called thermoclines. These thermoclines can oscillate in response to external perturbations, such as iceberg calving or water mixing, creating dynamic changes in the ocean profile.

The Brunt–Väisälä frequency ($N$), also known as the buoyancy frequency, is a key parameter that governs the stability of the ocean’s stratification. It describes the frequency at which a parcel of water will oscillate when displaced vertically, and it is essential for understanding the behavior of internal waves and thermoclines. The Brunt–Väisälä frequency is given by the equation:

$$N = \sqrt{\left(\frac{-g}{\rho_0} \frac{d\rho}{dz} \right)}$$

where: 


- $N$ is the Brunt–Väisälä frequency,
- $g$ is the acceleration due to gravity,
- $\rho_0$ is the reference density of seawater,
- $$\frac{d\rho}{dz}$$ is is the vertical density gradient.

​This frequency is crucial because it sets a limit on the vertical movement of thermoclines: if the oscillation frequency of the thermocline exceeds $N$, the oscillations will decay rather than propagate, and the thermocline will become unstable.

In this project, we use the DAS and DTS data from the Greenland ice sheet survey (Fig. 1) to extract the corner frequencies corresponding to the thermoclines. By analyzing the spectral content of the temperature signals from DTS and correlating them with the DAS data, we can identify the frequencies at which thermoclines oscillate. This frequency is essential for understanding the stability of these layers in the ocean and for inferring the ocean’s density profile. Using the Brunt–Väisälä equation, we can also estimate the density gradient and the corresponding stratification of the ocean water near the Greenland ice sheet, providing insights into the ocean's vertical structure and its response to external perturbations.

![AOI](https://github.com/Benz-Poobua/Internal-tsunami-revealed-by-DAS/blob/8bb1ff522567416941931def67034ee1e376d780/Figures/AOI.png)
Figure 1. The area of interest (AOI) where the survey was conducted. The figure above shows aerial imagery with the cable represented by a yellow line. The blue section highlights the area of interest, corresponding to the cable distance from 2829 m to 3150 m. This segment of the cable is nearly vertical and provides a reasonable representation of the ocean profile.

## 2. Methods
### Step 0:  Metadata Exploration
In this project, we use Distributed Acoustic Sensing (DAS) and Distributed Temperature Sensing (DTS) data to investigate internal waves in the ocean near the Greenland ice sheet. The data is acquired through the Sintela ONYX interrogator, with metadata related to both the interrogator and the cable stored in the h5 metadata attributes. The DAS system is configured with a gauge length of 4.79 meters, a sampling rate of 1 Hz, and a total of 1,587 loci (channels). To analyze the data, we slice it both in time and space, focusing on the section of the cable where the water is entered at approximately 560 meters and exits at 7,565 meters. Specifically, we start our data slicing at a distance of 2820 meters (near the east ridge) and end at 3150 meters (the seafloor, approximately 3150 meters in optical distance onward).

### Step 1: DAS visualization in Fig. 2 (`DAS_visualization.ipynb`)
1. Read the data using `read_Onyx_h5_to_list` 
2. Combine the files using `comb_Onyx_data`
- Combine data from files with identical attributes
3. Fill continuous parts of data into the array using `split_continuous_data`
- Separate parts of data where the time difference is greater than the pulse rate (1 Hz) 
- Store such continuous data in lists
4. Fill data gaps in the array using `fill_data_gaps` 
- Fill gaps with NaN
5. Filter DAS data using `scipy.signal.butter`
- Apply `scipy.signal.sosfiltfilt` to data 
6. Fill data gaps using `fill_data_gaps`
7. Convert to datetime using `sintela_to_datetime`
- These h5 files contain raw data: phase and timestamp. The timestamp is in Unix epoch format, so it needs to be converted to datetime first. 
![das](https://github.com/Benz-Poobua/Internal-tsunami-revealed-by-DAS/blob/8bb1ff522567416941931def67034ee1e376d780/Figures/das.png)
Figure 2. DAS visualization

### Step 2: DTS visualization in Fig. 3 (`DTS_visualization.ipynb`)
1. Read the DTS data (CSV format)
2. The columns represent datetime, and the rows indicate the  distance of the cable.
![dts](https://github.com/Benz-Poobua/Internal-tsunami-revealed-by-DAS/blob/8bb1ff522567416941931def67034ee1e376d780/Figures/dts.png)
Figure 3. DTS visualization. There are three layers representing different temperature zones. The interfaces are thermoclines.

### Step 3:  Construct `pick_internalwaves.py`
1. The function allows users to track the phase signals in DAS. The program will store point data (phase and datetime) in a CSV file. 
2. The program displays both DAS and DTS windows for a clear illustration. 
![window](https://github.com/Benz-Poobua/Internal-tsunami-revealed-by-DAS/blob/8bb1ff522567416941931def67034ee1e376d780/Figures/13_14_l.png)
Figure 4. The working window of `pick_internalwave.py`. When the cursor is clicked, the data (indicated by the red dot) is collected and saved in a CSV file, including the phase and datetime. These data points are then used to construct the time series of the buoyancy frequency by calculating the spectrogram.

### Step 4: Construct a time series of the buoyant frequencies 
1. Each CSV file in the 1-day data collection will be concatenated.
2. This concatenated file will be resampled and interpolated to fill gaps using mean values. 
3. `CubicSpline` is used to interpolate data before analyzing the frequency content. 
4. We calculate corner frequencies that may represent buoyancy frequencies by analyzing the frequency content in three distinct ranges, partitioned at 1 hour and 5 minutes.
1) Low-Frequency Range (< 1 hour):
- We compute the mean amplitude within this range, resulting in a horizontal line that represents the mean amplitude.
2) Mid-Frequency Range (1 hour to 5 minutes):
- We perform a linear regression on this range to derive a line that captures the frequency-dependent trend of the amplitudes.
3) High-Frequency Range (> 5 minutes):
- The high-frequency content is disregarded, as it is not relevant to internal wave signals.
![freq](https://github.com/Benz-Poobua/Internal-tsunami-revealed-by-DAS/blob/8bb1ff522567416941931def67034ee1e376d780/Figures/freq.png)
Figure 5. Calculation of corner frequencies representing buoyancy frequencies by analyzing the frequency content in three distinct ranges. In the low-frequency range (< 1 hour), the mean amplitude is computed, resulting in a horizontal line. In the mid-frequency range (1 hour to 5 minutes), a linear regression is performed to capture the frequency-dependent trend of the amplitudes. The high-frequency content (> 5 minutes) is disregarded as it is not relevant to internal wave signals.

## 3. Results
The intersection point of the horizontal line (low-frequency range) and the regression line (mid-frequency range) is identified as the corner frequency, which is then stored as a representation of the buoyancy frequency.
![time](https://github.com/Benz-Poobua/Internal-tsunami-revealed-by-DAS/blob/8bb1ff522567416941931def67034ee1e376d780/Figures/time.png)
Figure 6. Time series of the corner frequency. The upper plot shows the frequency on a logarithmic scale, while the lower plot depicts the corresponding period, calculated as 
10^log(Frequency). Red lines indicate arbitrary reference frequencies or periods (5 minutes and 1 hour) associated with internal waves, as identified in previous surveys (Dominik Gräff). The green line represents the trend of the buoyancy frequency (or period). Frequencies or periods within the defined boundaries are considered valid for analysis.





