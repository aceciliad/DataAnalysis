# DataAnalysis

## Overview
The scripts included here are designed to do a quick processing of the InSight seismic data.

Everything here is simplified for fast use. 

For basic preprocesing applied to the traces, catalog reading, etc., the reader is refered to the codes employed by the [Marsquake Service](https://github.com/sstaehler/mqs_reports/).

Two preprocessed events are included in `Data`. For more, please request the author. 

## Table of Contents
- [filter_banks_specgram.py](#filter_banks_specgram.py)
- [polarization_analysis.py](#polarization_analysis.py)


## filter_banks_specgram.py

This script produces a figure with a fast analysis of the waveforms corresponding to the selected event and component. 

| Option          | Description                                                 |
|-----------------|-------------------------------------------------------------|
| `--e`        | Use this flag to select the event to plot (e.g., S0235b).|
| `--c`        | Use this flag to select the component to plot (e.g., Z). Components available are Z, N, E, R, T. When it's available, it will use the backazimuth provided in the Events_picks_new.npy file (MQS unless it was modified) to rotate to R or T. In case it's not available, it will request a value from the user.|
| `--baz`        | Use this flag to select the backazimuth used to rotate to R or T. Event though a MQS backazimuth may be available, it will force the script to use the user value.| 
| `--save`        | Use this flag to save the output. If not present, saving is disabled. |
| `-h`, `--help`  | Displays help information about the script.                 |

While most flags are selected as plotting options, the `--save` flag is linked to the function `save_picks`  where a seismic phase `phase` (e.g., PP) and a delta time `dt` (e.g., 18.2) need to be defined to update the file `Events_picks_new.npy` where the information of the events is saved.

Please note that these scripts have been developed with the goal of picking seismic phases (i.e., following Duran et al., [2022](https://www.sciencedirect.com/science/article/pii/S0031920122000127)).


### Usage example

The following provides an example on how to run the script to provide the waveform analysis for the R component, considering a backazimuth of 44.14 degrees (Posiolova et al., [2022](https://www.science.org/doi/10.1126/science.abq7704)).

```bash
python filter_banks_specgram.py --e S1094b --c R --baz 44.14
````

The script was originally designed to pick seismic phases. In case of wanting to mark a new phase (or refining an old one), the function `save_picks` can be modified (see parameters `phase` and `dt` to add a new phase pick), and the following can be run:

```bash
python filter_banks_specgram.py --e S1094b --c R --baz 44.14 --save 

# Example output:
moving phase PP to 18 sec. from reference time
Are you sure you want to save the file? (y/n): y
# to ensure we do not mess up, it will only overwrite the file when we input "y"
````

### Other custom options

Among others the following parameters can be easily modified inside the script to depict different charasteristics of the signal:

- `tnorm0`: when plotting the data, traces are normalized in a time window selected to depict clearly the desired phases. The start of this normalization window is given by `tnorm0` as time in seconds with respect to the reference time (`self.tref_utc`), which is typically the arrival time for P wave.
- `tnorm1`: when plotting the data, traces are normalized in a time window selected to depict clearly the desired phases. The end of this normalization window is given by `tnorm1` as time in seconds with respect to the reference time (`self.tref_utc`), which is typically the arrival time for P wave.
- `data_dir`: can be modified to run with denoised data (see Dahmen et al., [2024](https://doi.org/10.1093/gji/ggae279) for details and denoised dataset).
- `fmin`, `fmax`: the values set in the `plot_event` function can be modified to adapt the bandpass filter to the desired frequency band.
- `vmin_spe`, `vmax_spe`: clipping values for the scalogram.
- `polarized`: in the function `plot_spectrogram` the parameter `polarized` can be set to True or False to produce the spectrogram of the polarized or non polarized waveforms, respectively.


## polarization_analysis.py

This script provides a fast analysis of the polarization attributes of the signal. Following Zenhaeusern et al., ([2022](https://pubs.geoscienceworld.org/ssa/bssa/article/112/4/1787/613988/Low-Frequency-Marsquakes-and-Where-to-Find-Them)), we compute the polarization attributes in order to plot the 3-components scalogram, the particle motion and a density estimate of the particle motion. 

![See figure](https://github.com/aceciliad/DataAnalysis/blob/main/Figures/PolAnalysis_S0235b.pdf)

Nevertheless, the polarization analysis includes other values that can also be employed to analyze particle motion such as:
- `inclination`
- `ellipticity`




