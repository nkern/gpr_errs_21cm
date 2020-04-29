import numpy as np
import hera_cal as hc
from uvtools import dspec
import hera_pspec as hp

def qe(x1, freqs, x2=None, window=None, R=None, cov=None, beam=None, scalar=None,
       el=0, eh=0, tavg=True, blavg=False, cross=False):
    """
    Form quadratic estimator and compute errors and window function.
        p_a = x.T E_a x
        E_a = 1/2 M_ab q_b = 1/2 M_ab R Q_b R 
        Q_b = fft(I)_b fft(I)_b.T
        V = 2tr[C E_a C E_b]
        W = M H
        H = 1/2 tr[E_a Q_b] = 1/2 tr[R Q_a R Q_b]
        M_ab ~ m_aa * delta_ab
        M_aa = 1 / sum_b[H_ab]

    Normalization assumes diagonal M

    Parameters
    x1 : DataContainer with ndarray (Ntimes, Nfreqs) [Jy]
    freqs : ndarray, frequency array [Hz]
    x2 : DataContainer with ndarray, optional. Default is to use x1
    window : str, windowing function
    R : Datacontainer with ndarray of weighting matrices if window is specified,
        it is multiplied along the diagonal
    cov : DataContainer with ndarray, to compute errorbars (Nfreqs, Nfreqs)
    beam : PSpecBeam object, for normalizing the power spectra
    scalar : float, pspec normalization scalar X^2Y
    el : int, edgecut_low
    eh : int, edgecut_hi
    tavg : bool, if True, time average power spectra
    blavg : bool, if True, average all output pspec keys
    cross : bool, if True, take cross spectra of adjacent time records

    Returns
    p : DataContainer, pspectra [mK^2 Mpc^3]
    pcov : DataContainer, pspectra covariance
    w : DataContainer, pspectra window functions
    dlys : ndarray, delay array
    """
    Nfreqs = len(freqs)
    Nbls = len(x1)
    win = dspec.gen_window(window, Nfreqs, edgecut_low=el, edgecut_hi=eh)

    # get X^2Y/Opp scalar
    if beam is not None:
        if isinstance(beam, str):
            beam = hp.PSpecBeamUV(beam)
        # get X^2Y
        if scalar is None:
            scalar = beam.compute_pspec_scalar(freqs[0], freqs[-1], Nfreqs, taper=window, exact_norm=True)
        Opp = beam.get_Omegas([('pI', 'pI')])[1]
        # divide by Opp
        scalar /= Opp[np.argmin(np.abs(beam.beam_freqs - freqs.mean())), 0]
        # incorporate Jy2mK factor
        scalar *= beam.Jy_to_mK(freqs)**2
    else:
        scalar = 1.0

    # form dC/dp matrix of shape (Nfreqs, Nfreqs, Nfreqs)
    Q = np.array([np.fft.fftshift(np.fft.ifft(np.eye(Nfreqs), axis=-1), axes=-1)[i] for i in range(Nfreqs)])
    Q = np.array([q[None, :].T.conj().dot(q[None, :]) for q in Q])

    pspec = hc.datacontainer.DataContainer({})
    pcov = hc.datacontainer.DataContainer({})
    for k in x1:
        # get R
        if R is None:
            r = np.eye(Nfreqs)
        else:
            r = R[k]
        r = r * win

        # compute H and M
        H = 0.5 * np.array([[np.trace(r.dot(Qa).dot(r).dot(Qb)) for Qb in Q] for Qa in Q])
        M = np.eye(Nfreqs) / np.sum(H, axis=-1) * scalar

        # compute E: shortcut knowing M is diagonal
        E = 0.5 * M.diagonal() * r.dot(Q[np.arange(Nfreqs)]).dot(r)

        # compute fft(R x) : use ifft instead of Qa for speed
        x1fft = np.fft.fftshift(np.fft.ifft(r.dot(x1[k]), axis=-1), axes=-1)
        if x2 is None:
            x2fft = x1fft
        else:
            x2fft = np.fft.fftshift(np.fft.ifft(r.dot(x2[k]), axis=-1), axes=-1)

        # compute q = 0.5 * fft(R x)^T fft(R x)
        if cross:
            q = 0.5 * x1fft[::2].conj() * x2fft[1::2]
        else:
            q = 0.5 * x1fft.conj() * x2fft

        # normalize
        p = M.dot(q)

        # time average
        if tavg:
            pspec[k] = np.mean(pspec[k], axis=0, keepdims=True)

        # calculate bandpower covariance
        if cov is not None:
            pcov[k] = 2 * np.array([[]])


            u, s, v = np.linalg.svd(cov[k])
            L = u.dot(np.eye(len(s)) * np.sqrt(s))
            Lfft = np.fft.fftshift(np.fft.ifft(L * win[:, None], axis=0), axes=0)
            #neb = hc.vis_clean.noise_eq_bandwidth(win)
            pcov[k] = 2 * Lfft.dot(Lfft.T.conj())**2 * scalar**2
            #Q = np.fft.ifft(np.eye(Nfreqs), axis=1)
            #pcov[k] = np.zeros((Nfreqs, Nfreqs), np.complex)
            #for i in range(Nfreqs):
            #    for j in range(Nfreqs):
            #        Qa = Q[i][:, None].dot(Q[:, None].T.conj())
            #        Qb = Q[i][:, None].dot(Q[:, None].T.conj())
            #        pcov[k][i, j] = 2 * np.trace(Qa.dot(cov[k]).dot(Qb).dot(cov[k]))

    dlys = np.fft.fftshift(np.fft.fftfreq(Nfreqs, np.diff(freqs)[0])) * 1e9
    if blavg:
        # take uniform average and get error on uniform average
        k = list(pspec.keys())[0]
        ps = np.mean([pspec[k] for k in pspec], axis=0)
        pc = np.sum([pcov[k] for k in pcov], axis=0) / Nbls**2
        pspec = hc.datacontainer.DataContainer({k: ps})
        pcov = hc.datacontainer.DataContainer({k: pc})

    return pspec, pcov, dlys




