#!/usr/bin/env python
# -*- coding: utf-8 -*-

#==========================================================================
# Run filter banks
#==========================================================================

# third party imports
import numpy as np
import os, argparse, datetime
from obspy.core import UTCDateTime, read

# plotting stuff
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator
from cmcrameri import cm as cm_cra
import matplotlib.patheffects as pe

# project specific imports
import polarization_filter as pf
from utils import *

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

    ap.add_argument('--c', action='store', dest='component',
                    default='Z',
                    help='component', type=str)

    return ap.parse_args()


class PLOT_WAVEFORMS:

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

        #import ipdb; ipdb.set_trace()  # noqa

        #self.tref_utc   = UTCDateTime(2021, 12, 24, 22, 45, 10, 666041)
        #self.tswave_utc = UTCDateTime(2021, 12, 24, 22, 50, 58, 149623)

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


    def plot_event(self,*,
                   fmin=0.2,
                   fmax=0.9):

        self.fmin   = fmin
        self.fmax   = fmax
        fig, ax_st, ax_pz, ax_fb    = self.set_figure()

        self.plot_waveforms(ax_st)
        self.plot_waveforms(ax_pz,
                            polarized=True)

        self.plot_filterbanks(ax_fb)

        self.set_labels(ax_st, ax_pz, ax_fb)


        os.makedirs(figs_dir, exist_ok=True)
        plt.savefig(os.path.join(figs_dir,
                                 f'FilterBanks_{self.event_id}_{self.component}.pdf'))

        return


    def plot_waveforms(self, ax,
                       win_env=5.,
                       win_pol=5.,
                       polarized=False):

        kwargs  = {'color':'grey',
                   'lw':0.5,
                   'ls':'solid',
                   'zorder':10}

        stream_copy = self.stream_raw.copy()
        stream_copy.filter(type='bandpass',
                          freqmin=self.fmin,
                          freqmax=self.fmax,
                          corners=4,
                          zerophase=True)

        stream_copy.trim(starttime=self.tref_utc+ax.get_xlim()[0]-100,
                         endtime=self.tref_utc+ax.get_xlim()[1]+100)

        if polarized:
            stream_copy = pf.polarization_filter(stream_copy,
                                                 win_pol)

        stream_norm = stream_copy.copy()
        stream_norm.trim(starttime=self.tref_utc+ax.get_xlim()[0],
                         endtime=self.tref_utc+ax.get_xlim()[1])

        max_val = np.max(np.abs(stream_norm.select(component=self.component)[0].data))
        ax.set_ylim([-1.5*max_val, 1.5*max_val])

        tr  = stream_copy.select(component=self.component)[0]
        ax.plot(tr.times(reftime=self.tref_utc),
                tr.data,
                **kwargs)

        tr_env  = envelope_smooth(tr=tr,
                                  mode='same',
                                  envelope_window_in_sec=win_env)

        ax.plot(tr_env.times(reftime=self.tref_utc),
                tr_env.data*2,
                color='#E07A5F',
                zorder=11,
                alpha=1, lw=0.7,
                rasterized=False)

        return


    def plot_filterbanks(self,ax,*,
                         f0_filter= 1./5.7,
                         f1_filter= 1.,
                         df=2.**0.5,
                         win_pol=5.,
                         win_env=5):

        nfreqs  = int(np.round(np.log(f1_filter / f0_filter) / np.log(df), decimals=0) + 1)
        freqs   = np.geomspace(f0_filter, f1_filter + 0.001, nfreqs)

        cmap_p  = cm_cra.batlow
        colors  = [cmap_p(val) for val in np.linspace(0.,0.8,nfreqs)]

        dir_poli   = os.path.join(data_dir, 'FilterBanks')
        os.makedirs(dir_poli,
                    exist_ok=True)

        for ifreq, fcenter in enumerate(freqs):
            ifreq2  = np.copy(ifreq)
            f0 = fcenter / df
            f1 = fcenter * df

            stream_copy  = self.stream_raw.copy()
            # trimming it makes it faster to compute polarized waveforms
            stream_copy.trim(starttime=self.tref_utc+ax.get_xlim()[0]-100.,
                             endtime=self.tref_utc+ax.get_xlim()[1]+100.)
            stream_copy.taper(max_percentage=0.1)
            stream_copy.filter('bandpass',
                              freqmin=f0, freqmax=f1,
                              corners=4, zerophase=True)

            tr  = stream_copy.select(component=self.component)[0]
            tr, norm    = norm_trace(tr,
                                     reftime=self.tref_utc,
                                     tstart=ax.get_xlim()[0],
                                     tend=ax.get_xlim()[1])
            tr.data *= 0.5
            tr_env  = envelope_smooth(tr=tr, mode='same',
                                      envelope_window_in_sec=win_env)

            # For ploting set glitches manually to nans
            #tr.data[tr.data>.7]=np.nan
            #tr.data[tr.data<-.5]=np.nan
            #tr_env.data[tr_env.data>.7]=np.nan

            ax.plot(tr.times(reftime=self.tref_utc),
                    tr.data+ifreq+.3,
                    color='lightgrey',
                    zorder=7, lw=.7)

            ax.plot(tr_env.times(reftime=self.tref_utc),
                    tr_env.data+ifreq+.3,
                    color='grey',
                    zorder=9, lw=.7)

            # Polarized waveforms
            freq_id = str(fcenter).replace('.','-')[:4]
            fname_poli  = f'FB_{self.event_id}_{freq_id}.mseed'
            path_poli   = os.path.join(dir_poli,
                                       fname_poli)

            try:
                stream_poli = read(path_poli)
            except:
                stream_copy2  = self.stream_raw.copy()
                # trimming it makes it faster to compute polarized waveforms
                stream_copy2.trim(starttime=self.tref_utc+ax.get_xlim()[0]-100.,
                                 endtime=self.tref_utc+ax.get_xlim()[1]+100.)
                stream_copy2.taper(max_percentage=0.1)
                stream_copy2.filter('bandpass',
                                  freqmin=f0, freqmax=f1,
                                  corners=4, zerophase=True)

                stream_poli = pf.polarization_filter(stream_copy2,
                                                     win_pol)

                stream_poli.write(path_poli,
                                  format='MSEED')

            tr_poli     = stream_poli.select(component=self.component)[0]
            tr_poli, norm  = norm_trace(tr_poli,
                                     reftime=self.tref_utc,
                                     tstart=ax.get_xlim()[0],
                                     tend=ax.get_xlim()[1])
            tr_poli.data    *= 0.5
            tr_env  = envelope_smooth(tr=tr_poli, mode='same',
                                      envelope_window_in_sec=win_env)

            if ifreq==len(freqs)-1:
                color = colors[ifreq]
                color = '#e2cfc9'
                ax.text(-5, ifreq+0.35, 'Raw',
                        color='gray', zorder=30, ha='right',
                        path_effects=[pe.withStroke(linewidth=1.5,
                                         foreground="white")])
                ax.text(-5, ifreq-0.05, 'Polarized',
                        color='#ccaa9f', zorder=30,va='top', ha='right',
                        path_effects=[pe.withStroke(linewidth=1.5,
                                         foreground="white")])
            else:
                lightness=1.4
                saturation=0.4
                color=adjust_color(colors[ifreq],
                                   lightness=lightness,
                                   saturation=saturation)

            # Set up glitches to nan manually
            #tr_poli.data[tr_poli.data>0.5]    = np.nan
            #tr_poli.data[tr_poli.data<-0.5] = np.nan
            #tr_env.data[tr_env.data>.5]=np.nan
            #tr_env.data[tr_env.data<-.5]=np.nan

            ax.plot(tr_poli.times(reftime=self.tref_utc),
                    tr_poli.data+ifreq,
                    color=color,
                    #color=lighten_color(colors[ifreq], amount=0.5),
                    zorder=9, lw=0.5)

            ax.plot(tr_env.times(reftime=self.tref_utc),
                    tr_env.data+ifreq, color=colors[ifreq],
                    zorder=11, lw=.7)


        ticklabels = []
        for freq in freqs:
            if freq > 1:
                ticklabels.append(f'{freq:.1f}')
            else:
                ticklabels.append(f'1/{1. / freq:.1f}')

        ax.set_yticks(np.arange(nfreqs))
        ax.set_ylim([-0.5, nfreqs-0.2])

        ax.set_yticklabels(ticklabels)


        return







    def set_figure(self):


        fig = plt.figure()
        cm  = 1/2.54
        fig.set_size_inches(10*cm,17*cm)

        gs  = gridspec.GridSpec(ncols=1, nrows=2,
                                bottom=0.65, top=0.96,
                                left=0.16, right=0.97,
                                figure=fig,
                                wspace=0.12, hspace=.2)

        ax_st   = fig.add_subplot(gs[0])
        ax_pz   = fig.add_subplot(gs[1])

        for ax in [ax_st, ax_pz]:
            ax.spines[['right', 'top', 'bottom']].set_visible(False)
            ax.set_xlim([-100, 850])
            ax.tick_params(labelbottom=False, bottom=False)
            ax.set_ylabel('Amplitude\n(m/s)', labelpad=-0.2)

        gs_fb   = gridspec.GridSpec(ncols=1, nrows=1,
                                    bottom=0.06,
                                    top=gs.bottom-0.02,
                                    left=gs.left,
                                    right=gs.right,
                                    figure=fig,
                                    wspace=gs.wspace, hspace=.3)

        ax_fb   = fig.add_subplot(gs_fb[:])

        ax_fb.spines[['right', 'top']].set_visible(False)
        ax_fb.set_xlim(ax_st.get_xlim())
        ax_fb.set_xlabel('Time (s)')
        ax_fb.xaxis.set_major_locator(MultipleLocator(250.))
        ax_fb.xaxis.set_minor_locator(MultipleLocator(50.))
        ax_fb.set_ylabel('Central frequency (Hz)')

        return fig, ax_st, ax_pz, ax_fb


    def set_labels(self, ax_st, ax_pz, ax_fb):

        for ax in [ax_st, ax_pz, ax_fb]:
            for tt, pha in zip([0, self.tswave_utc-self.tref_utc],
                               [self.pha_ref, self.pha_swave]):
                ax.vlines(tt, -10, 10,
                          color='#3D405B',
                          ls='dashed',
                          lw=0.7,
                          zorder=20)
                if ax in [ax_st]:
                    ax.text(tt+3, ax.get_ylim()[1], pha,
                            color='#3D405B',
                            ha='left', va='top')

        ax_st.text(ax_st.get_xlim()[1], ax_st.get_ylim()[1],
                   f'{self.component}',
                   zorder=100, va='top',
                   ha='right',clip_on=False,
                   bbox= dict(boxstyle='square',
                               facecolor='#ecdab5',
                               edgecolor='none',
                               alpha=0.7))

        return



if __name__=='__main__':

    results = arguments()
    event   = results.event
    comp    = results.component

    plot_obj    = PLOT_WAVEFORMS(data_dir=data_dir,
                                 event_id=event,
                                 component=comp)

    plot_obj.read_catalog()
    plot_obj.get_waveforms()
    plot_obj.plot_event()


