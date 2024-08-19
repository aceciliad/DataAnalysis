#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Non-linear time domain polarization filter. No warranty implied.
Developed for InSight mission to Mars.

Author: Savas Ceylan - ETH Zurich
Changes/Additions: Cecilia Duran
Copyright 2020 by ETH Zurich

The polarization filter code is attached. This will smooth out the elliptical signals to enhance body-waves.

import pzfilter
pzfilter.pzfilter (data, nwin, component_order)

data can be a 3xN array, a list of three traces, or a stream.

Because streams can have an arbitrary number of traces which are not in order, you should tell the code what to use in the component_order, e.g. 'ZRT'. You can leave it as None, then the code will first try ZRT, and then ZNE if it fails. Data needs to be pre-processed as you wish, and no gaps are allowed.

nwin is the window length (number of samples).

The output will be polarization filtered waveforms (3xN array in the same order as the input), rectilinearity, directivity and prinpical axis.

"""
import numpy as np
import obspy


def pzfilter(waveforms, nwin, channel_order=None):
    """
    Time-domain polarization filtering. This is a re-implementation of
    Matlab codes from Quancheng Huang (University of Maryland), received
    via personal communication.

    Method is from:
    Montalbetti and Kanasewich (1970), doi:10.1111/j.1365-246X.1970.tb01771.x

    :type waveforms: obspy.Stream, list, or numpy array
    :param waveforms: Pre-processed three component seismic data. Note
    that the theory is explained in the ZRT framework. It can be of
    type obspy.Stream, a list of three obspy.Trace objects, or a
    3xN matrix. If a Stream or Trace list is given, the data is prepared
    implicitly looking for ZRT components first, and ZNE second.
    The component order can be specified using the 'channels' option.
    All artifacts such as data gaps must be taken care of in advance.

    :type nwin: int
    :param nwin: Window length (number of samples)

    :type channel_order: iterable
    :param channel_order: Optional. Used only if the 'waveforms'
    parameter is an obspy.Stream instance. Specifies in which order the
    components should be structured into a matrix form. If not specified,
    first ZRT then ZNE channels will be collected from the 'waveforms'.
    If all fails, a ValueException is raised.

    :rtype tuple
    :return: The filtered waveforms as 3xN array in the same order,
    smoothed rectilinearity, smoothed directivity for three components,
    eigenvectors of the principal axis.
    """
    def smooth(data, winlen, method='mean', mode='same'):
        """ Smooth and return the data using its mean or sum. """
        _supported_methods = ['mean', 'sum']
        if method not in _supported_methods:
            raise ValueError('Method not supported. Use one of ' +
                             str(_supported_methods))

        w = np.ones(int(winlen), )

        if method.lower() == 'mean':
            w /= int(winlen)
        elif method.lower() == 'sum':
            w /= w.sum()

        data = np.convolve(data, w, mode=mode)

        return data


    def validate_data(data):
        """ Check the data """
        if len(data) == 0:
            _err = "Inconsistent data: Pass a Stream, a list of three " \
                   "Traces, or a 3xN shaped matrix. Data lengths must be " \
                   "the same"
            raise ValueError(_err)

        if not all(len(elem) == len(data[0]) for elem in data):
            raise ValueError('Traces must have the same number of samples')


    def prepare(data, component_order=None):
        """ Prepare the data. """
        _data_for_polarization = []

        # A stream is passed
        if isinstance(data, obspy.Stream):
            if component_order is None:
                try:
                    component_order = ['*Z', '*R', '*T']
                    for _channel in component_order:
                        _data_for_polarization.append(
                            data.select(channel=_channel)[0].data)
                except:
                    try:
                        component_order = ['*Z', '*N', '*E']
                        for _channel in component_order:
                            _data_for_polarization.append(
                                data.select(channel=_channel)[0].data)
                    except:
                        raise ValueError('Stream does not have any '
                                         'known channels (ZNE or ZRT)')
            else:
                if len(component_order) != 3:
                    raise ValueError('Provide 3 components in channel_order')
                try:
                    for _channel in component_order:
                        if not _channel.startswith('*'):
                            _channel = '*' + _channel
                        _data_for_polarization.append(
                            data.select(channel=_channel)[0].data)
                except:
                    raise ValueError('At least one component cannot be '
                                     'found in the Stream')

        # A list is passed, either with arrays of Traces.
        elif isinstance(data, list) or isinstance(data, np.ndarray):
            # Check the shape
            if len(data) != 3:
                raise ValueError('Input must be in 3xN matrix form')

            if all(isinstance(_val, obspy.Trace) for _val in data):
                # When the list contains obspy.Trace instances
                _data_for_polarization = [_tr.data for _tr in data]
            elif all(isinstance(_val, (list, np.ndarray)) for _val in data):
                # When the list contains another numpy array or list
                _data_for_polarization = [_d for _d in data]
            else:
                # Something else is passed
                _err = "Inconsistent data: Pass a Stream, a list of three " \
                       "Traces, or a 3xN shaped matrix. Data lengths must " \
                       "be the same"
                raise ValueError(_err)

        validate_data(_data_for_polarization)

        return _data_for_polarization

    # ------------------------------------------------------------------
    # Main method starts here
    # ------------------------------------------------------------------
    # Empirical constants: may change to be more strict with filtering
    J = 1.0
    K = 2.0
    n = 0.5
    # Collect the data and validate
    S   = prepare(data=waveforms, component_order=channel_order)

    npts    = len(S[0])
    nwin2   = int(round(nwin / 2))

    # Pad the data with zeros at both ends.
    S = np.concatenate([np.zeros((nwin2, 3), float),
                        np.transpose(S),
                        np.zeros((nwin2, 3), float)])

    # Compute polarization
    rectilinearity  = []
    directivity     = []
    principal_axis  = []

    for i in np.arange(nwin2, nwin2 + npts):
        # Window the data, compute eigen values/vectors from its
        # covariance matrix
        S_cut = S[np.arange((i - nwin2), (i + nwin2 + 1)), :]
        V = np.cov(S_cut, rowvar=False)
        eigen_val, eigen_vec = np.linalg.eig(V)

        # Sort the eigen values in reverse order along with the
        # corresponding eigen vectors
        idx = eigen_val.argsort()[::-1]
        idx = idx[::-1]
        eigval = eigen_val[idx]
        eigvec = eigen_vec[:, idx]

        # Compute rectilinearity (F) with empirical regularization
        # factor (J) applied
        # Output can be modified to save non-polarized part
        F = 1-(eigval[1] / eigval[2]) ** n
        F = F ** J

        # Store
        rectilinearity.append(F)
        principal_axis.append(np.transpose(eigvec[:, 2]))
        directivity.append(np.transpose(eigvec[:, 2]) ** K)

    # Switch to numpy arrays for better functionality
    rectilinearity = np.asarray(rectilinearity)
    rectilinearity[np.isnan(rectilinearity)] = 0.0
    #rectilinearity[rectilinearity<0.2] = 0.
    principal_axis = np.asarray(principal_axis)
    directivity = np.asarray(directivity)

    # Smooth the rectilinearity and directivity
    rectilin_smooth = smooth(rectilinearity, nwin / 2)
    directivity_smooth = [smooth(directivity[:, 0], nwin/2),
                          smooth(directivity[:, 1], nwin/2),
                          smooth(directivity[:, 2], nwin/2)]
    directivity_smooth = np.transpose(directivity_smooth)

    # Apply the smoothed polarization filter
    S_filtered = np.nan * np.zeros((npts, 3))
    for i in np.arange(0, npts - 1):
        S_filtered[i, :] = np.multiply(S[i + nwin2, :],
                                       np.multiply(rectilin_smooth[i],
                                                   directivity_smooth[i, :]))

    return (np.transpose(S_filtered), rectilin_smooth, directivity_smooth,
            principal_axis)



def polarization_filter(stream, lwin=5,
                        delta=None):
    '''
    apply polarization filter
    '''
    comp    = []
    for tr in stream:
        comp.append(tr.stats.component)

    print(f'    > Applying polarization filter')

    if not delta:
        nwin        = int(lwin/stream[0].stats.delta)
    else:
        nwin        = int(lwin/delta)

    st_pol      = stream.copy()
    data_pol    = pzfilter(stream, nwin, ('').join(comp))

    # Assign data to stream
    stream_pol  = stream.copy()

    for i in range(3):
        stream_pol[i].data[:-1] = data_pol[0][i][:-1]

    return stream_pol


