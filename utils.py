#!/usr/bin/env python
# -*- coding: utf-8 -*-

#==========================================================================
# Some extra utility functions
#==========================================================================

import numpy as np
import datetime, os, argparse, sys
from obspy.clients.fdsn import Client
from obspy import UTCDateTime, read, read_inventory
from obspy.signal.filter import envelope
from matplotlib.colors import LinearSegmentedColormap
from obspy.signal.tf_misfit import cwt
from obspy.signal.util import next_pow_2
import matplotlib as mpl

sys.path.insert(0, 'polarization_package')
import polarisation_calculation as pc


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


def cmap_pa():
    cmap_colors    = ['blue', 'cornflowerblue', 'goldenrod', 'gold', 'yellow',
                    'darkgreen', 'green', 'mediumseagreen', 'darkred',
                    'firebrick','tomato', 'midnightblue', 'blue']
    custom_cmap =  LinearSegmentedColormap.from_list('custom',
                                                     cmap_colors,
                                                     N=len(cmap_colors))
    custom_cmap.set_bad('white',1.)

    return custom_cmap



def calc_cwf(tr,
             tref,
             fmin=0.001,
             fmax=1,
             w0=24):
    dt = tr.stats.delta

    scalogram = abs(cwt(tr.data, dt,
                        w0=w0, nf=150,
                        fmin=fmin, fmax=fmax))

    def create_timearray(tr):
        timevec = [t+(tr.stats.starttime-tref) for t in tr.times()]
        return timevec

    t = create_timearray(tr)
    f = np.logspace(np.log10(fmin),
                    np.log10(fmax),
                    scalogram.shape[0])

    return scalogram ** 2, f, t


def save_polarized_data(stream, output_name, tstart=None, tend=None, system='ZNE',
                        differentiate=False, winlen_sec=10.,
                        fmin=0.1, fmax=2., dop_specwidth=1.1,
                        dop_winlen=10., kind='cwt', nf=250,
                        overlap=0.5, detick_1Hz=False, w0=8,
                        use_alpha=True, use_alpha2=False,
                        alpha_inc=None, alpha_elli=1., alpha_azi=None):
    '''
    save data of azimuth, ellipticity and inclination computed using polarization package
    Modified from Zenhäusern et al., (2022)
    '''

    # differentiate waveforms: apply in case stream is in DISP
    if differentiate:
        stream.differentiate()

    dict_comp   = {'ZNE':['Z', 'N', 'E'],
                   'ZRT':['Z', 'R', 'T']}
    comps       = dict_comp[system]

    st_Z, st_N, st_E    = [stream.select(component=cc) for cc in comps]

    if tstart is None:
        tstart  = max([st.stats.starttime for st in stream])
    if tend is None:
        tend    = min([st.stats.endtime for st in stream])

    tstart, tend, dt = pc._check_traces(st_Z, st_N, st_E, tstart, tend)


    winlen_samp = int(winlen_sec / dt)
    nfft        = next_pow_2(winlen_samp) * 2

    # Calculate width of smoothing windows for degree of polarization analysis
    nfsum, ntsum, dsfacf, dsfact = pc._calc_dop_windows(
        dop_specwidth, dop_winlen, dt, fmax, fmin,
        kind, nf, nfft, overlap, winlen_sec)


    # Detick data for SEIS data
    for tr_Z, tr_N, tr_E in zip(st_Z, st_N, st_E):
        if tr_Z.stats.npts < winlen_samp * 4:
            continue

        if detick_1Hz:

            tr_Z_detick, tr_N_detick, tr_E_detick   = [pc.detick(st_ii,10) for st_ii in [tr_Z, tr_N, tr_E]]

            f, t, u1, u2, u3 = pc._compute_spec(tr_Z_detick, tr_N_detick, tr_E_detick,
                                                kind, fmin, fmax, winlen_samp,
                                                nfft, overlap,
                                                nf=nf, w0=w0)
        else:
            f, t, u1, u2, u3 = pc._compute_spec(tr_Z, tr_N, tr_E, kind,
                                                fmin, fmax, winlen_samp,
                                                nfft, overlap, nf=nf, w0=w0)

        azi1, azi2, elli, inc1, inc2, r1, r2, P = pc.compute_polarization(
            u1, u2, u3, ntsum=ntsum, nfsum=nfsum, dsfacf=dsfacf, dsfact=dsfact)

    f = f[::dsfacf]
    t = t[::dsfact]
    time    = np.repeat(tr_Z.stats.starttime, len(t)) + t

    #Scalogram and alpha/masking of signals
    scalogram = 10 * np.log10((r1 ** 2).sum(axis=-1))
    # alpha, alpha2 = polarization._dop_elli_to_alpha(P, elli, use_alpha, use_alpha2)
    r1_sum, alpha, alpha2 = polarisation_filtering(r1, inc1, azi1, azi2, elli,
                                                   alpha_inc, alpha_azi, alpha_elli,
                                                   P)

    scalogram_filt  = 10 * np.log10(r1_sum)

    data    = {'freq': f,
               'time': time,
               'alpha': alpha,
               'azimuth': np.rad2deg(azi1),
               'ellipticity': elli,
               'inclination': np.rad2deg(abs(inc1)),
               'scalogram': scalogram,
               'scalogram_filt':scalogram_filt,
               'r1':r1}

    np.save(output_name, data)

    return


#def polarisation_filtering(r1, inc1, azi1, azi2, elli,
#                           alpha_inc, alpha_azi, alpha_elli,
#                           P,use_alpha, use_alpha2 ):
#    '''
#    modified from Zenhäusern et al., (2022)
#    '''
#    if alpha_inc is not None:
#        if alpha_inc > 0.: #S
#            func_inc= np.cos
#            func_azi= np.sin
#            # func_azi= np.cos #S0173a special filtering for Cecilia
#        else: #P
#            alpha_inc= -alpha_inc
#            func_inc= np.sin
#            func_azi= np.cos
#            # func_azi= np.sin #S0173a special filtering for Cecilia
#    else:
#        #look at azimuth without inclination, let's just set it like this.
#        #So cosinus prefers P waves, set to sinus to prefer S waves (perpendicular to BAZ)
#        func_azi= np.cos
#
#    r1_sum = (r1** 2).sum(axis=-1)
#    if alpha_inc is not None:
#        r1_sum *= func_inc(inc1)**(2*alpha_inc)
#    if alpha_azi is not None:
#        r1_sum *= abs(func_azi(azi1))**(2*alpha_azi)
#    if alpha_elli is not None:
#        r1_sum *= (1. - elli)**(2*alpha_elli)
#
#    alpha, alpha2= pc._dop_elli_to_alpha(P, elli, use_alpha, use_alpha2)
#
#    if alpha_inc is not None:
#        alpha*= func_inc(inc1)**alpha_inc
#    if alpha_azi is not None:
#        alpha*= abs(func_azi(azi1))**alpha_azi
#    if alpha_elli is not None:
#        alpha*= (1. - elli)**alpha_elli
#
#    return r1_sum, alpha, alpha2


def polarisation_filtering(r1, inc1, azi1, azi2, elli,
                           alpha_inc, alpha_azi, alpha_elli,
                           P):
    # Apply filtering based on degree of polarisation (dop) and possibly ellipticity/inclination/azimuth
    # alpha_azi only really makes sense with ZRT/LQT data
    if alpha_inc is not None:
        if alpha_inc > 0.: #When looking for S
            func_inc= np.cos
            func_azi= np.sin
        else: #When looking for P
            alpha_inc= -alpha_inc
            func_inc= np.sin
            func_azi= np.cos
    else:
        #look at azimuth without inclination, let's just set it like this.
        #So cosinus prefers P waves, set to sinus to prefer S waves (perpendicular to BAZ)
        func_azi= np.cos

    r1_sum = (r1** 2).sum(axis=-1)
    if alpha_inc is not None:
        r1_sum *= func_inc(inc1)**(2*alpha_inc)
    if alpha_azi is not None:
        r1_sum *= abs(func_azi(azi1))**(2*alpha_azi)
    if alpha_elli is not None:
        r1_sum *= (1. - elli)**(2*alpha_elli)

    alpha, alpha2= pc._dop_elli_to_alpha(P, elli)

    if alpha_inc is not None:
        alpha*= func_inc(inc1)**alpha_inc
    if alpha_azi is not None:
        alpha*= abs(func_azi(azi1))**alpha_azi
    if alpha_elli is not None:
        alpha*= (1. - elli)**alpha_elli

    return r1_sum, alpha, alpha2



def pcolormesh_alpha(ax, x, y, val, alpha, vmin,
                     vmax, cmap=None, bounds=None):
    '''
    modified from Zenhäusern et al., (2022)
    '''

    if cmap is None:
        cmap_colors    = ['blue', 'cornflowerblue', 'goldenrod', 'gold', 'yellow',
           'darkgreen', 'green', 'mediumseagreen', 'darkred', 'firebrick',
           'tomato', 'midnightblue', 'blue']
        cmap =  LinearSegmentedColormap.from_list('custom', cmap_colors, N=len(cmap_colors))
        #custom_cmap.set_bad('white',1.)

    if bounds is None:
        qm = ax.pcolormesh(x, y, val[:-1,:-1], vmin=vmin, vmax=vmax,
                           cmap=cmap, linewidth=0., rasterized=True) #, shading='flat' is implicitly used, which drops last column+row of val. Raises deprecation warning, so doing it manually

        colors = qm.cmap(qm.norm(qm.get_array()))
        norm = qm.norm
    else: #for custom colorscale
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        qm = ax.pcolormesh(x, y, val[:-1,:-1], cmap=cmap, norm=norm,
                           linewidth=0., rasterized=True)
        colors = qm.cmap(qm.norm(qm.get_array()))

    # if val.shape = (len(x), len(y)), then pcolormesh neglects one column +
    # row in val, hence need to adapt alpha
    colors = colors.reshape(-1, 4)  # this is not needed in earlier python versions
    colors[:, -1] = alpha[:len(y)-1, :len(x)-1].ravel()
    qm.set_color(colors)

    # create scalar mappable for colorbar
    cm = mpl.pyplot.cm.ScalarMappable(cmap=qm.cmap, norm=norm)
    cm.set_array([])

    # if the mappable array is not none, the colors are recomputed on draw from
    # the mappable
    qm._A = None
    return


def find_nearest(arr, val):
    idx = np.abs(arr-val).argmin()
    return arr[idx], idx

