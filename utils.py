#!/usr/bin/env python
# -*- coding: utf-8 -*-

#==========================================================================
# Some extra utility functions
#==========================================================================

import numpy as np
import datetime, os, argparse
from obspy.clients.fdsn import Client
from obspy import UTCDateTime, read, read_inventory
from obspy.signal.filter import envelope


def envelope_smooth(envelope_window_in_sec,
                    tr,
                    mode='valid'):

    tr_env = tr.copy()
    tr_env.data = envelope(tr_env.data)

    w = np.ones(int(envelope_window_in_sec / tr.stats.delta))
    w /= w.sum()
    tr_env.data = np.convolve(tr_env.data, w, mode=mode)

    return tr_env


def norm_trace(tr,
               reftime,
               tstart=0,
               tend=500):

    times   = tr.times(reftime=reftime)

    idx0    = np.argwhere(times>tstart)[0][0]
    idx1    = np.argwhere(times>tend)[0][0]

    data_sel    = tr.data[idx0:idx1]

    norm_val= np.nanmax(np.abs(data_sel))
    tr.data /= norm_val

    return tr, norm_val


def adjust_color(color, lightness=1.2, saturation=0.5):
    """
    Adjusts the given color by modifying its lightness and saturation.
    Arguments:
    - color: a tuple with three elements (r, g, b) or four elements (r, g, b, a)
    - lightness: a float that defines the factor by which to lighten the color (1 means no change)
    - saturation: a float that defines the factor by which to adjust the saturation (1 means no change)
    Returns:
    - a tuple with the new color values in RGB(A) format
    """
    import matplotlib.colors as mc
    import colorsys

    try:
        c = mc.cnames[color]  # If it's a named color
    except:
        c = color
    r, g, b, *a = c
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    l = min(1, max(0, lightness * l))
    s = min(1, max(0, saturation * s))
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    if a:
        return (r, g, b, a[0])
    return (r, g, b)


