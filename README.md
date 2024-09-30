# DataAnalysis

## Overview
The scripts included here are designed to do a quick processing of the InSight seismic data.

Everything here is simplified for fast use. 

For basic preprocesing applied to the traces, catalog reading, etc., the reader is refered to the github employ by the Marsquake Service (https://github.com/sstaehler/mqs_reports/).

## Table of Contents
- [filter_banks_specgram.py](#filter_banks_specgram.py)
- [Usage](#usage)
- [Options](#options)
- [Examples](#examples)



## filter_banks_specgram.py

This script produces a figure with a fast analysis of the waveforms corresponding to the event selected. 

| Option          | Description                                                 |
|-----------------|-------------------------------------------------------------|
| `--e`        | Use this flag to select the event to plot (e.g., S0235b).|
| `--c`        | Use this flag to select the component to plot (e.g., Z). Components available are Z, N, E, R, T. When it's available, it will use the backazimuth provided in the Events_picks_new.npy file (MQS unless it was modified) to rotate to R or T. In case it's not available, it will request a value from the user.|
| `--baz`        | Use this flag to select the backazimuth used to rotate to R or T. Event though a MQS backazimuth may be available, it will force the script to use the user value.| 
| `--save`        | Use this flag to save the output. If not present, saving is disabled. |
| `-h`, `--help`  | Displays help information about the script.                 |



