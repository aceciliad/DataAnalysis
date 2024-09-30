# DataAnalysis

## Overview
The scripts included here are designed to do a quick processing of the InSight seismic data.

Everything here is simplified for fast use. 

For basic preprocesing applied to the traces, catalog reading, etc., the reader is refered to the github employ by the Marsquake Service (https://github.com/sstaehler/mqs_reports/).

Two preprocessed events are included in "Data". For more, request the author. 

## Table of Contents
- [filter_banks_specgram.py](#filter_banks_specgram.py)



## filter_banks_specgram.py

This script produces a figure with a fast analysis of the waveforms corresponding to the event selected. 

| Option          | Description                                                 |
|-----------------|-------------------------------------------------------------|
| `--e`        | Use this flag to select the event to plot (e.g., S0235b).|
| `--c`        | Use this flag to select the component to plot (e.g., Z). Components available are Z, N, E, R, T. When it's available, it will use the backazimuth provided in the Events_picks_new.npy file (MQS unless it was modified) to rotate to R or T. In case it's not available, it will request a value from the user.|
| `--baz`        | Use this flag to select the backazimuth used to rotate to R or T. Event though a MQS backazimuth may be available, it will force the script to use the user value.| 
| `--save`        | Use this flag to save the output. If not present, saving is disabled. |
| `-h`, `--help`  | Displays help information about the script.                 |

While most flags are selected as plotting options, the `--save` flag is linked to the function `save_picks`  where a phase (e.g., PP) and a delta time `dt` (e.g., 18.2) need to be selected to update the file `Events_picks_new.npy` where the information of the events is saved.


## Usage example

```bash
python filter_banks_specgram.py --e S1094b --c R --baz 44.14
````

In case of wanting to mark a new phase (or refining an old one), the function `save_picks` can be modified, and the following can be run:

```bash
python filter_banks_specgram.py --e S1094b --c R --baz 44.14 --save 

# Example output:
moving phase PP to 18 sec. from reference time
Are you sure you want to save the file? (y/n): y
# and will only overwrite the file when "y" is input into the terminal
````

## Other custom options

Among others the following parameters can be easily modified inside the script to depict different charasteristic of the signal.

- `tnorm0`: when plotting the data, traces are normalized in a time window selected to depict clearly the desired phases. The start of this normalization window is given by `tnorm0`, which is the time in seconds with respect to the reference time (`self.tref_utc`), which is typically the arrival time for P wave.
- `tnorm1`: when plotting the data, traces are normalized in a time window selected to depict clearly the desired phases. The end of this normalization window is given by `tnorm1`, which is the time in seconds with respect to the reference time (`self.tref_utc`), which is typically the arrival time for P wave.
