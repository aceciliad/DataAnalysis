#!/usr/bin/env python
# -*- coding: utf-8 -*-

#==========================================================================
# Run filter banks
#==========================================================================

# third party imports
import numpy as np
import os, argparse, datetime
from obspy.core import UTCDateTime, read
from collections import OrderedDict

# plotting stuff
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator
from cmcrameri import cm as cm_cra
import matplotlib.patheffects as pe
from matplotlib.ticker import FormatStrFormatter

# project specific imports
import polarization_filter as pf
from utils import *

# global params
mpl.rcParams.update({'font.size': 8})
#data_dir    = 'Data'
data_dir    = 'DenoisedEvents'
figs_dir    = 'Figures'
tnorm0  = 400.
tnorm1   = 600.


#==========================================================================
def arguments():
    '''
    arguments
    '''
    ap = argparse.ArgumentParser(description='Plot event')

    ap.add_argument('--c', action='store', dest='component',
                    default='Z',
                    help='component', type=str)

    return ap.parse_args()


class PLOT_WAVEFORMS:

    def __init__(self, *,
                 data_dir,
                 component='Z'):

        self.data_dir   = data_dir
        self.component  = component

        return


    def read_catalog(self,*,
                     ts_min=160.,
                     ts_max=190.,
                     quality=['A']):
        '''
        normally I would read the station catalog (.xml file) to get
        the default picks and event info for the desired marsquake
        (uncertainties, distance, etc.)
        '''

        picks_events    = np.load(os.path.join(data_dir,
                                               'Events_picks.npy'),
                                  allow_pickle=True).item()

        self.picks  = {}

        for event, picks in picks_events.items():
            # go through all events and only save the ones with ts_min<ts<ts_max

            tref    = picks.get('P',
                                picks.get('PP',
                                          picks.get('x1',
                                                    picks.get('Pg',picks.get('start')))))
            tref_utc   = UTCDateTime(tref)
            pha_ref    = [i for i in picks if picks[i]==tref_utc][0]

            tswave  = picks.get('S',
                                picks.get('SS',
                                          picks.get('x2',
                                                    picks.get('Sg',picks.get('start')))))
            tswave_utc = UTCDateTime(tswave)
            pha_swave  = [i for i in picks if picks[i]==tswave_utc][0]

            baz_mqs    = picks['baz']

            if (tswave_utc-tref_utc)<=ts_max and (tswave_utc-tref_utc)>=ts_min:
                if picks['quality'] in quality:
                    info_ev = {event:{'tref_utc':tref_utc,
                                      'pha_ref':pha_ref,
                                      'tswave_utc':tswave_utc,
                                      'pha_swave':pha_swave,
                                      'ts':tswave_utc-tref_utc,
                                      'baz':baz_mqs}}

                    self.picks.update(info_ev)

        # sort events by Ts-Tp
        self.picks  = dict(sorted(self.picks.items(), key=lambda item: item[1]['ts']))

        return


    def get_waveforms(self,*,
                      event_id,
                      picks,
                      type_stream='VEL',
                      fmin=0.05,
                      fmax=10.):
        '''
        here normally I would download the traces from IRIS Client
        service, using the catalog properties, and preprocess them...
        '''
        self.stream_raw = read(os.path.join(self.data_dir,
                                            f'*{event_id}*'))

        if self.component in ['R', 'T']:

            baz_mqs = picks['baz']
            print(f' Using MQS backazimuth {baz_mqs}')
            backazimuth = baz_mqs

            if not backazimuth:
                print(f' Backazimuth not available for event {event_id}  to rotate to {self.component} component')
                backazimuth = float(input(f' Introduce custom backazimuth for event {event_id}: '))

            self.stream_raw.rotate(method='NE->RT',
                                   back_azimuth=backazimuth)

        return


    def plot_event(self,*,
                   fmin=0.4,
                   fmax=0.7):

        self.fmin   = fmin
        self.fmax   = fmax
        fig, ax = self.set_figure()

        self.plot_waveforms(ax,
                            polarized=True)

        os.makedirs(figs_dir, exist_ok=True)
        plt.savefig(os.path.join(figs_dir,
                                 f'ComparisonEvents_{self.component}.pdf'))

        return


    def plot_waveforms(self, ax,
                       win_env=5.,
                       win_pol=5.,
                       polarized=False):

        kwargs  = {'color':'darkgrey',
                   'lw':0.5,
                   'ls':'solid',
                   'zorder':10}

        ax.set_ylim([-0.5, len(self.picks)-0.5])

        ticks   = []

        for ii, (event, picks) in enumerate(self.picks.items()):

            self.get_waveforms(event_id=event,
                               picks=picks)

            stream_copy = self.stream_raw.copy()
            stream_copy.filter(type='bandpass',
                              freqmin=self.fmin,
                              freqmax=self.fmax,
                              corners=4,
                              zerophase=True)

            stream_copy.trim(starttime=picks['tref_utc']+ax.get_xlim()[0]-100,
                             endtime=picks['tref_utc']+ax.get_xlim()[1]+100)

            if self.fmax<2:
                stream_copy.resample(sampling_rate=4*self.fmax, window='hann')

            if polarized:
                stream_copy = pf.polarization_filter(stream_copy,
                                                     win_pol)

            stream_norm = stream_copy.copy()
            stream_norm.trim(starttime=picks['tref_utc']+tnorm0,
                             endtime=picks['tref_utc']+tnorm1)

            tr_norm = stream_norm.select(component=self.component)[0]
            max_val = np.nanmax(np.abs(tr_norm.data))*3

            tr  = stream_copy.select(component=self.component)[0]

            tr_env  = envelope_smooth(tr=tr,
                                      mode='same',
                                      envelope_window_in_sec=win_env)

            # put nans to daba above max_val
            tr.data[np.abs(tr.data)>max_val]  = np.nan
            tr_env.data[np.abs(tr_env.data)>max_val]  = np.nan

            ax.plot(tr.times(reftime=picks['tref_utc']),
                    ii+tr.data/max_val,
                    **kwargs)

            ax.plot(tr_env.times(reftime=picks['tref_utc']),
                    ii+(tr_env.data/max_val)*1.5,
                    color='#be4d4d',
                    zorder=11,
                    alpha=1, lw=0.7,
                    rasterized=False)

            ax.text(ax.get_xlim()[1]+3, ii, event,
                    color='dimgrey', va='center')

            ticks.append(picks['ts'])

            ax.vlines([picks['ts'], picks['ts']],
                      ymin=ii-0.3, ymax=ii+0.3,
                      color='#679666',
                      ls='solid', lw=1.,
                      zorder=20)

        ax.set_yticks(np.arange(0, len(self.picks)))
        ticks_fmt   = [f"{value:.1f}" for value in ticks]
        ax.set_yticklabels(ticks_fmt)

        ax.vlines([0,0], ymin=-10, ymax=100,
                  color='#679666',
                  ls='dashed', lw=1.,
                  zorder=20)

        ax.text(0, ax.get_ylim()[1], 'P',
                color='#679666',
                ha='center', va='bottom')

        ax.text(ticks[-1], ax.get_ylim()[1], 'S',
                color='#679666',
                ha='center', va='bottom')

        ax.grid(visible=True,
                axis='both',
                which='both',
                color='dimgray',
                ls='dotted',
                lw=0.5)


        return


    def set_figure(self):

        fig = plt.figure()
        cm  = 1/2.54
        fig.set_size_inches(13*cm,15*cm)

        gs  = gridspec.GridSpec(ncols=1, nrows=1,
                                bottom=0.07, top=0.98,
                                left=0.12, right=0.9,
                                figure=fig,
                                wspace=0.12, hspace=.2)

        ax  = fig.add_subplot(gs[0])

        ax.spines.top.set_visible(False)
        ax.spines.right.set_visible(False)
        ax.set_xlim([-100, 750])

        ax.set_ylabel(r'$\rm{T_S}$-$\rm{T_P}$ (s)')
        ax.set_xlabel('Time (s)')
        ax.xaxis.set_major_locator(MultipleLocator(250.))
        ax.xaxis.set_minor_locator(MultipleLocator(50.))

        return fig, ax



if __name__=='__main__':

    results = arguments()
    comp    = results.component

    plot_obj    = PLOT_WAVEFORMS(data_dir=data_dir,
                                 component=comp)

    plot_obj.read_catalog()
    plot_obj.plot_event()


