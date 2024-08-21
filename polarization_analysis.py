#!/usr/bin/env python
# -*- coding: utf-8 -*-

#==========================================================================
# Run polarization analysis
# Uses the Polarisation analysis for seismic data published in
# https://zenodo.org/records/7220543 by Zenhausern et al., (2022)
#==========================================================================

# third party imports
import numpy as np
import os, argparse, datetime, sys
from obspy.core import UTCDateTime, read, Stream
from obspy.signal.util import next_pow_2
from scipy import stats

# plotting stuff
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator
from cmcrameri import cm as cm_cra
import matplotlib.patheffects as pe
from matplotlib import mlab as mlab

# project specific imports
from utils import *

sys.path.insert(0, 'polarization_package')
import polarisation_calculation as pc

# global params
mpl.rcParams.update({'font.size': 8})
data_dir    = 'Data'
figs_dir    = 'Figures'

#==========================================================================
def arguments():
    '''
    arguments
    '''
    ap = argparse.ArgumentParser(description='Plot event')

    ap.add_argument('--e', action='store', dest='event',
                    required=True,
                    help='event name', type=str)

    return ap.parse_args()



class PLOT_POLANALYSIS:

    def __init__(self, *,
                 data_dir,
                 event_id,
                 component='Z'):
        self.data_dir   = data_dir
        self.event_id   = event_id
        self.component  = component

        return


    def read_catalog(self):
        '''
        normally I would read the station catalog (.xml file) to get
        the default picks and event info for the desired marsquake
        (uncertainties, distance, etc.)
        '''

        picks_events    = np.load(os.path.join(data_dir,
                                               'Events_picks.npy'),
                                  allow_pickle=True).item()

        picks   = picks_events[self.event_id]

        tref    = picks.get('P',
                            picks.get('PP',
                                      picks.get('x1',
                                                picks.get('Pg',picks.get('start')))))
        self.tref_utc   = UTCDateTime(tref)
        self.pha_ref    = [i for i in picks if picks[i]==self.tref_utc][0]


        tswave  = picks.get('S',
                            picks.get('SS',
                                      picks.get('x2',
                                                picks.get('Sg',picks.get('start')))))
        self.tswave_utc = UTCDateTime(tswave)
        self.pha_swave  = [i for i in picks if picks[i]==self.tswave_utc][0]

        return


    def get_waveforms(self,*,
                     type_stream='VEL',
                     fmin=0.05,
                     fmax=10.):
        '''
        here normally I would download the traces from IRIS Client
        service, using the catalog properties, and preprocess them...
        '''

        self.stream_raw = read(os.path.join(self.data_dir,
                                            f'*{self.event_id}*'))

        return


    def plot_analysis(self):

        xwin    = np.array([-100, self.tswave_utc-self.tref_utc+200])
        fig, ax_azi, ax_spe, ax_kde, ax_den = self.set_fig(xwin=xwin)

        self.plot_scalogram(ax_spe)

        self.plot_azimuth(ax_azi, ax_kde, ax_spe, ax_den)


        os.makedirs(figs_dir, exist_ok=True)
        plt.savefig(os.path.join(figs_dir,
                                 f'PolAnalysis_{self.event_id}.pdf'))

        return


    def plot_azimuth(self, ax_azi, ax_kde, ax_spe, ax_den):

        data_pol    = self.get_pa()

        specgram    = data_pol['scalogram']
        time        = data_pol['time']
        freq_sp     = data_pol['freq']
        azimuth     = data_pol['azimuth']
        alpha       = data_pol['alpha']
        inclination = data_pol['inclination']

        time_arr    = (time-self.tref_utc).astype('float64')

        pcolormesh_alpha(ax_azi, time_arr, freq_sp, azimuth,
                         alpha=alpha, vmin=0., vmax=360.)

        phases  = [self.pha_ref, self.pha_swave]
        freqs   = [0.2, 0.8]
        times   = [0., self.tswave_utc-self.tref_utc]
        colors  = ['grey', '#542a3e']  #['#cc3b3a', '#542a3e']
        dtt = 5. # width of window to compute KDE, it could also be the uncertainty

        azi_grid= np.linspace(0., 360., 100)
        max_azi = []

        for pha, tt, color in zip(phases, times, colors):

            fa, fb  = freqs
            fa_val, fa_idx = find_nearest(freq_sp, fa)
            fb_val, fb_idx = find_nearest(freq_sp, fb)

            azimuth_fb  = azimuth[fa_idx:fb_idx, :]

            # select time range
            ta_val, ta_idx = find_nearest(time_arr, tt-dtt)
            tb_val, tb_idx = find_nearest(time_arr, tt+dtt)

            azimuth_tt  = azimuth_fb[:, ta_idx:tb_idx].ravel()

            kde     = stats.gaussian_kde(azimuth_tt,
                                         bw_method=0.2)
            azi_kde = kde.evaluate(azi_grid)
            azi_kde /= np.max(azi_kde)

            ax_kde.plot(azi_grid, azi_kde,
                        color=color, lw=1.5,
                        zorder=4)

            max_val = azi_grid[np.argmax(azi_kde)]
            max_azi.append(max_val)

            ypos    = ax_kde.get_ylim()[1]
            diff    = np.diff(max_azi)

            if len(diff) > 0 and np.abs(diff) < 120.:
                ypos = 1.1

            ax_kde.vlines(max_val,
                          ax_kde.get_ylim()[0],
                          ypos-0.1,
                          colors=color,
                          lw=1., zorder=5, alpha=0.7,
                          ls='solid')
            ax_kde.text(max_val, ypos,
                        '{:5.1f}°'.format(max_val), color=color,
                        ha='center', va='top', zorder=10,
                        path_effects=[pe.withStroke(linewidth=1.5,
                                                 foreground="white")])

            # mark phases in azimuth plot
            for ax in [ax_azi, ax_spe, ax_den]:
                ax.vlines(tt, ymin=ax.get_ylim()[0],
                          ymax=ax.get_ylim()[1],
                          colors=color,
                          lw=2,
                          zorder=100)

            ax_spe.text(tt, ax_spe.get_ylim()[1],
                        pha, color=color,
                        va='bottom',
                        ha='center',
                        fontweight='bold')

        # Now noise
        fa_val, fa_idx = find_nearest(freq_sp, fa)
        fb_val, fb_idx = find_nearest(freq_sp, fb)

        azimuth_fb  = azimuth[fa_idx:fb_idx, :]

        # select time range
        ta_val, ta_idx = find_nearest(time_arr, -100)  #-50
        tb_val, tb_idx = find_nearest(time_arr, -50)   #-20

        azimuth_tt  = azimuth_fb[:, ta_idx:tb_idx].ravel()

        kde     = stats.gaussian_kde(azimuth_tt,
                                     bw_method=0.2)
        azi_kde = kde.evaluate(azi_grid)
        azi_kde /= np.max(azi_kde)

        ax_kde.fill_between(azi_grid, 0., azi_kde,
                            color='grey', ec=None,
                            alpha=0.5, zorder=1)


        # KDE azimuth in time
        n_azi   = 100
        azi_grid  = np.linspace(0., 180., n_azi)

        azimuth2    = np.copy(azimuth)
        azimuth2    = np.where(azimuth2<180., azimuth2, azimuth2-180.)
        azimuth_den = azimuth2[fa_idx:fb_idx, :]
        time_ref    = time - self.tref_utc

        # moving time window
        ntime   = 1000
        win     = ax_spe.get_xlim()
        time_grid   = np.linspace(win[0]-50, win[1]+50, ntime)
        time_width  = 10.

        azi_time    = np.empty((n_azi, ntime))
        azi_time[:] = np.nan

        max_azis    = np.array(())
        for ii, ti in enumerate(time_grid):

            ta_val, ta_idx = find_nearest(time_ref, ti)
            tb_val, tb_idx = find_nearest(time_ref, ti+time_width)

            azimuth_win = azimuth_den[:, ta_idx:tb_idx]
            azi_plot = azimuth_win.ravel()
            kde = stats.gaussian_kde(azi_plot, bw_method=0.2)

            azi_kde = kde.evaluate(azi_grid)
            azi_kde /= np.max(azi_kde)

            max_azis    = np.append(max_azis, azi_grid[np.argmax(azi_kde)])
            azi_time[:,ii] = azi_kde

        ax_den.pcolormesh(time_grid+5, azi_grid, azi_time,
                          norm=self.norm_den,
                          rasterized=True, cmap=self.cmap_den)

        return


    def get_pa(self,
               pa_folder='Data/Polarization_Analysis'):

        stream  = self.stream_raw.copy()
        tref    = self.tref_utc

        os.makedirs(pa_folder,
                    exist_ok=True)
        data_pa = os.path.join(pa_folder,
                               f'PA_analysis_{self.event_id}')

        if len(stream)>3:
            stream  = stream.merge()

        stream  = self.stream_raw.copy()
        stream.trim(starttime=self.tref_utc-200.,
                    endtime=self.tswave_utc+500.)
        stream.filter(type='bandpass',
                      freqmin=0.01,
                      freqmax=2,
                      corners=4,
                      zerophase=True)
        stream.resample(sampling_rate=4, window='hann')

        stream_detick   = Stream()
        for ii, tr in enumerate(stream):
            tr = pc.detick(tr,
                           detick_nfsamp=5)

            stream_detick.append(tr)

        stream  = stream_detick.copy()

        try:
            data_pol= np.load(data_pa+'.npy', allow_pickle=True).item()
            return data_pol
        except:
            print('     Apply polarization analysis')
            save_polarized_data(stream=stream, system='ZNE',
                                output_name=data_pa,
                                kind='cwt', fmin=0.04, fmax=5.,
                                winlen_sec=10., overlap=0.5,
                                dop_winlen=8, dop_specwidth=1.1,
                                nf=100, w0=8,
                                use_alpha=True, use_alpha2=False,
                                alpha_inc = None, alpha_elli = None,
                                alpha_azi = None, #None when not used
                                differentiate = False, detick_1Hz = False)

            data_pol= np.load(data_pa+'.npy', allow_pickle=True).item()

            return data_pol



    def plot_scalogram(self, ax):
        stream  = self.stream_raw.copy()
        stream.trim(starttime=self.tref_utc-200.,
                    endtime=self.tswave_utc+500.)
        stream.filter(type='bandpass',
                      freqmin=0.01,
                      freqmax=2,
                      corners=4,
                      zerophase=True)
        stream.resample(sampling_rate=4, window='hann')

        SS  = []

        for ii, tr in enumerate(stream):
            tr = pc.detick(tr,
                           detick_nfsamp=5)

            Sxx, f, t = calc_cwf(tr=tr,
                                 tref=self.tref_utc,
                                 w0=6, fmax=6)
            #Sxx = scipy.ndimage.filters.gaussian_filter(Sxx,
            #                                           sigma=2, mode='constant')
            SS.append(Sxx)

        SS  = np.array(SS)
        freq_sp = np.copy(f)
        time_arr= np.copy(t)
        specgram    = 10* np.log10(SS.sum(axis=0))

        ax.pcolormesh(time_arr, freq_sp, specgram,
                      norm=self.norm_spe,
                      cmap=self.cmap_spe,
                      rasterized=True)

        return




    def set_fig(self,*,
                xwin=np.array([-100, 600])):

        fig = plt.figure()
        fig.set_size_inches(7.,5.5)

        gs  = gridspec.GridSpec(ncols=3, nrows=3, bottom=0.09,
                                top=0.93, left=0.09, right=0.94,
                                figure=fig, wspace=0.2, hspace=0.15)

        ax_spe  = fig.add_subplot(gs[0,:2])
        ax_azi  = fig.add_subplot(gs[1,:2])
        ax_den  = fig.add_subplot(gs[2,:2])

        ax_kde  = fig.add_subplot(gs[1,2])

        posw    = ax_kde.get_position()
        posh    = ax_spe.get_position()

        dpos    = 0.01
        gs_cb1  = gridspec.GridSpec(ncols=1, nrows=1,
                                    bottom=posh.y0+dpos,
                                    top=posh.y1-dpos,
                                    left=posw.x0,
                                    right=posw.x0+0.025,
                                    figure=fig)
        gs_cb2  = gridspec.GridSpec(ncols=1, nrows=1,
                                    bottom=posh.y0+dpos,
                                    top=posh.y1-dpos,
                                    left=posw.x1-0.045,
                                    right=posw.x1-0.02,
                                    figure=fig)

        posw    = ax_kde.get_position()
        posh    = ax_den.get_position()

        gs_cb3  = gridspec.GridSpec(ncols=1, nrows=1,
                                    bottom=posh.y0+dpos,
                                    top=posh.y1-dpos,
                                    left=posw.x0,
                                    right=posw.x0+0.025,
                                    figure=fig)


        ax_cbspe= fig.add_subplot(gs_cb1[0])
        ax_cbazi= fig.add_subplot(gs_cb2[0])
        ax_cbden= fig.add_subplot(gs_cb3[0])

        # set Spectrogram part
        ax_spe.set_ylim([0.1, 1])
        ax_spe.set_ylabel('Frequency (Hz)')

        ax_spe.tick_params(labelbottom=False)
        ax_spe.set_xlim(xwin)

        y_major = mpl.ticker.LogLocator(base = 10.0,
                                        numticks = 2)
        y_minor = mpl.ticker.LogLocator(base = 10.0,
                                        subs = np.arange(1.0, 10.0) * 0.1,
                                        numticks = 10)
        ax_spe.yaxis.set_major_locator(y_major)
        ax_spe.yaxis.set_minor_locator(y_minor)
        ax_spe.tick_params(which='minor', axis='y', labelleft=False)

        ax_spe.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
        ax_spe.set_yscale('log')

        ax_spe.xaxis.set_major_locator(MultipleLocator(200))
        ax_spe.xaxis.set_minor_locator(MultipleLocator(50))

        self.cmap_spe   = mpl.cm.plasma
        vmin_spe    = -210
        vmax_spe    = -150
        self.norm_spe    = mpl.colors.Normalize(vmin=vmin_spe,
                                                vmax=vmax_spe)

        cb = mpl.colorbar.ColorbarBase(ax_cbspe,
                                       orientation='vertical',
                                       cmap=self.cmap_spe,
                                       norm=self.norm_spe)
        cb.set_label(r'(m/s)$^{\rm{2}}$/Hz  (dB)')
        cb.ax.yaxis.set_major_locator(MultipleLocator(30))
        cb.ax.yaxis.set_minor_locator(MultipleLocator(10))

        # PA panel
        ax_azi.tick_params(labelbottom=False)
        ax_azi.xaxis.set_major_locator(ax_spe.xaxis.get_major_locator())
        ax_azi.xaxis.set_minor_locator(ax_spe.xaxis.get_minor_locator())
        ax_azi.set_ylabel('Frequency (Hz)')
        #ax_azi.set_xlabel('Time (s)')
        ax_azi.set_xlim(xwin)
        ax_azi.set_ylim(ax_spe.get_ylim())

        for ax in [ax_azi, ax_kde]:
            ax.xaxis.set_label_coords(0.5, -0.18)

        ax_azi.yaxis.set_major_locator(y_major)
        ax_azi.yaxis.set_minor_locator(y_minor)
        ax_azi.tick_params(which='minor', axis='y', labelleft=False)

        ax_azi.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())

        ax_azi.set_yscale('log')

        # Colorbar Azi
        cmap    = cmap_pa()
        norm_col= mpl.colors.Normalize(vmin=0, vmax=360)
        cb = mpl.colorbar.ColorbarBase(ax_cbazi, orientation='vertical',
                                       cmap=cmap, norm=norm_col)
        cb.set_label('Azimuth (deg)')

        ticks_cb    = np.arange(0,361, 90)
        labels_cb   = np.char.mod('%3.0f',ticks_cb)

        cb.set_ticks(ticks_cb)
        cb.ax.set_yticklabels(labels_cb)


        # KDE panel
        ax_kde.spines.right.set_visible(False)
        ax_kde.spines.top.set_visible(False)
        ax_kde.tick_params(left=False,
                           labelleft=False)
        ax_kde.set_xlim([0., 360])
        ax_kde.set_xlabel('Azimuth (deg)')
        ax_kde.set_ylim([0.,1.2])

        ax_kde.xaxis.set_major_locator(MultipleLocator(180))
        ax_kde.xaxis.set_minor_locator(MultipleLocator(45))

        fig.text(0.5, 0.99,
                f'{self.event_id}',
                zorder=100, va='top',
                ha='center',clip_on=False,
                bbox= dict(boxstyle='square',
                           facecolor='lightgray',
                           edgecolor='none',
                           alpha=0.7))

        # density azimuth
        ax_den.xaxis.set_major_locator(ax_spe.xaxis.get_major_locator())
        ax_den.xaxis.set_minor_locator(ax_spe.xaxis.get_minor_locator())
        ax_den.yaxis.set_major_locator(MultipleLocator(90))
        ax_den.yaxis.set_minor_locator(MultipleLocator(30))


        ax_den.set_ylabel('Azimuth (deg)')
        ax_den.set_xlabel('Time (s)')
        ax_den.set_xlim(xwin)
        ax_den.set_ylim([0, 180])

        # colorbar
        ncol    = len(mpl.cm.plasma.colors)
        colors_plas = [mpl.cm.plasma.colors[i] for i in range(0, ncol, 64)]
        colors_list = ['w', 'w', 'w'] + [mpl.cm.plasma.colors[0]]+ colors_plas

        self.cmap_den = LinearSegmentedColormap.from_list('mycmap',
                                                          colors_list,
                                                          N=200)
        # Colorbar
        self.norm_den = mpl.colors.Normalize(vmin=0, vmax=1)
        cb3 = mpl.colorbar.ColorbarBase(ax_cbden, orientation='vertical',
                                       cmap=self.cmap_den, norm=self.norm_den)
        cb3.set_label('Density', labelpad=-.3)

        ticks_cb = np.array([0., 1.])
        labels_cb = np.char.mod('%1.0f',ticks_cb)

        cb3.set_ticks(ticks_cb)
        cb3.ax.set_yticklabels(labels_cb)

        return fig, ax_azi, ax_spe, ax_kde, ax_den





if __name__=='__main__':

    results = arguments()
    event   = results.event

    plot_obj    = PLOT_POLANALYSIS(data_dir=data_dir,
                                   event_id=event)

    plot_obj.read_catalog()
    plot_obj.get_waveforms()
    plot_obj.plot_analysis()


