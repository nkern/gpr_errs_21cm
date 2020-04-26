"""
convencience functions for working with simulation data
"""
import numpy as np
import hera_cal as hc
from pyuvdata import UVData, utils as uvutils
import copy
from astropy.io import fits
from astropy import constants
import healpy
import functools
import operator


def setup_data(fg, eor, noise, fg2=None, eor2=None, noise2=None, bls=None,
               combine_reds=None, eor_tilt=None, noise_suppress=1.0, times=None, no_autos=True):
    """
    Setup data into hera_cal.FRFilter objects

    Parameters
    ----------
    fg : str or list of str, foreground files. Will sum together files in a list
    eor : str, eor file
    noise : str, noise file
    fg2 : str or list of str, interlelaved foreground files. Must match fg in shape
    eor2 : str, interleaved eor file, must match eor in shape
    noise2 : str, interleaved noise file, must match noise in shape
    bls : list, antpair tuples to load
    combine_reds : list of list of antpairs, antpairs to concatenate along time ax
    eor_tilt : float, spectral tilt to filter eor with
    noise_suppress : float, noise suppression factor
    times : int array, time indices to load (not values)
    no_autos : bool, if True, don't load autocorrs

    Returns
    -------
    summed object
    fg object
    eor object
    noise object
    """
    terms = []
    # foregrounds
    fg = configure_data(fg, bls=bls, times=times, no_autos=no_autos)
    if fg2 is not None:
        fg2 = configure_data(fg2, bls=bls, times=times, no_autos=no_autos)
        fg = interleave(fg, fg2)
    F = hc.frf.FRFilter(fg)
    terms.append(fg)
    # eor
    eor = load_data(eor, bls=bls, times=times, no_autos=no_autos)
    # change spectral tilt if desired
    if eor_tilt is not None:
        eor = eor_pl_tilt(eor, eor_tilt)
    if eor2 is not None:
        eor2 = load_data(eor2, bls=bls, times=times, no_autos=no_autos)
        if eor_tilt is not None:
            eor2 = eor_pl_tilt(eor2, eor_tilt)
        eor = interleave(eor, eor2)
    E = hc.frf.FRFilter(eor)
    terms.append(eor)
    # noise
    noise = load_data(noise, bls=bls, times=times, no_autos=no_autos)
    noise.data_array /= noise_suppress
    if noise2 is not None:
        noise2 = load_data(noise2, bls=bls, times=times, no_autos=no_autos)
        noise2.data_array /= noise_suppress
        noise = interleave(noise, noise2)
    N = hc.frf.FRFilter(noise)
    terms.append(noise)

    assert len(terms) > 0, "nothing fed"
    full = copy.deepcopy(terms[0])
    for t in terms[1:]:
        full.data_array += t.data_array
    D = hc.frf.FRFilter(full)

    # combine reds
    if combine_reds is not None:
        _combine_reds(D, combine_reds)
        _combine_reds(F, combine_reds)
        _combine_reds(E, combine_reds)
        _combine_reds(N, combine_reds)

    return D, F, E, N

def _combine_reds(D, reds):
    """
    """
    for red in reds:
        if len(red) == 1: continue
        for p in D.pols:
            D.data[red[0]+(p,)] = np.concatenate([D.data[bl+(p,)] for bl in red])
            D.flags[red[0]+(p,)] = np.concatenate([D.flags[bl+(p,)] for bl in red])
            for bl in red[1:]:
                del D.data[bl+(p,)]
                del D.flags[bl+(p,)]

def load_data(dfile, bls=None, times=None, no_autos=True):
    """load visibility file to UVData
    Parameters
    ----------
    dfile : str, path to uvdata
    bls : list, antpair tuples to load
    times : int ndarray, time indices (not values) to load
    no_autos: bool, if True don't load autocorrs
    """
    u = UVData()
    u.read(dfile, bls=bls)
    if no_autos:
        bls = u.get_antpairs()
        u.select(bls=[bl for bl in bls if bl[0] != bl[1]])
    if times is not None:
        u.select(times=np.unique(u.time_array)[times])
    return u

def configure_data(dfiles, bls=None, times=None, no_autos=True):
    """configure visibility data from file

    Parameters
    ----------
    dfiles : str, list of str
        if list of str:
            innermost list are loaded and summed
    bls : list, list of antpair tuples to load
    times : int array, time indices to load (not values)
    no_autos : bool, if True do not load autocorrelations

    Returns
    -------
    UVData
    """
    if isinstance(dfiles, (list, tuple)):
        uvds = []
        for i, df in enumerate(dfiles):
            uvds.append(load_data(df, bls=bls, times=times, no_autos=no_autos))
        data = [u.data_array for u in uvds]
        if len(data) > 1:
            uvds[0].data_array += functools.reduce(operator.add, data[1:])
        uvd = uvds[0]
    elif isinstance(dfiles, str):
        uvd = load_data(dfiles, bls=bls, times=times, no_autos=no_autos)
    else:
        uvd = dfiles

    return uvd


def interleave(uvd1, uvd2=None):
    if uvd2 is None:
        return uvd1
    sd = 86164.0908  # sidereal day in seconds
    times = np.unique(uvd1.time_array)
    lsts = np.unique(uvd1.lst_array)
    if len(times) == 1:
        dt = 1e-5
        dlst = dt * 0.997269569 * np.pi / 12
    else:
        dt = np.median(np.diff(times))
        dlst = np.median(np.diff(lsts))

    uvd2 = copy.deepcopy(uvd2)
    uvd2.time_array += dt / 2
    uvd2.lst_array = (uvd2.lst_array + dlst / 2) % (2*np.pi)

    return uvd1 + uvd2


def populate_point_sources(gleamfile, freqs, col='Fint151', cent_freq=151e6,
                           nside=128, minflux=1.0, afill=0):
    """populate healpix map with point sources in Kelvin
    
    Args:
    gleamfile : path to gleam fits file
    freqs : ndarray of frequencies Hz
    col : gleam column to use
    cent_freq : anchor freq for spix [Hz], default: 151e6
    nside : nside of healpix map
    minflux : minimum flux cut [Jy]
    afill : if spix (alpha) is None, fill with this value
    """
    # load
    gleam = fits.open(gleamfile)[1].data
    flux = gleam[col]
    scut = np.isfinite(flux)
    fcut = flux[scut] > minflux
    flux = flux[scut][fcut]
    ra = gleam['RAJ2000'][scut][fcut]
    dec = gleam['DEJ2000'][scut][fcut]
    alpha = gleam['alpha'][scut][fcut]
    Nsources = len(ra)

    # fill alpha
    alpha[~np.isfinite(alpha)] = afill

    # create spectra
    spectra = flux[None, :] * (freqs[:, None] / cent_freq)**alpha[None, :]

    # conver flux from Jy to Kelvin
    kb = constants.k_B.cgs.value
    lams = constants.c.cgs.value / freqs
    pxarea = healpy.nside2pixarea(nside, degrees=False)
    conversion = 1e-23 * lams**2 / 2 / kb / pxarea  # Kelvin/Jy
    spectra *= conversion[:, None]

    # create maps
    Nfreqs = len(freqs)
    npix = healpy.nside2npix(nside)
    m = np.zeros((Nfreqs, npix), np.float) 
    lon, lat = healpy.pix2ang(nside, np.arange(npix), lonlat=True)

    # iterate over sources and fill maps
    for i in range(Nsources):
        pix = np.argmin((lon-ra[i])**2+(lat-dec[i])**2)
        m[:, pix] += spectra[:, i]

    return m

def eor_pl_tilt(eor, tilt=0, anchor_dly=200):
    eor = copy.deepcopy(eor)
    efft = np.fft.fftshift(np.fft.fft(eor.data_array, axis=2), axes=2)
    dlys = np.fft.fftshift(np.fft.fftfreq(eor.Nfreqs, eor.channel_width)) * 1e9
    tilt = (np.abs(dlys).clip(50, np.inf) / anchor_dly)[None, None, :, None] ** tilt
    eor.data_array = np.fft.ifft(np.fft.ifftshift(efft * tilt, axes=2), axis=2)
    return eor


