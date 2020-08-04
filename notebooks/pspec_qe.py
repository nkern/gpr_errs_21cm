import numpy as np
import hera_cal as hc
from uvtools import dspec
import hera_pspec as hp
from hera_cal.datacontainer import DataContainer as DC

class QE:
    
    def __init__(self, freqs, x1, x2=None, C=None, scalar=None, cosmo=None):
        """
        Quadratic Estimator across freqs.
        
        uE = 0.5 * R Q R
        H_ab = tr(uE_a Q_b)
        q = x1 uE x2
        M = H^-1, H^-1/2, etc.
        E_a = M_ab uE_b
        p = M q = x1 E x2
        W = M H
        V = 2 tr(C E C E)
        b = tr(C E)
        
        Parameters
        ----------
        freqs : ndarray (Nfreqs)
            frequency array of x in MHz
        x1 : ndarray or hera_cal DataContainer (Ntimes, Nfreqs)
             Complex visibility data as left-hand input for QE
        x2 : ndarray or hera_cal DataContainer (Ntimes, Nfreqs)
             Complex visibility data as right-hand input for QE
             Default is x1
        C : ndarray (Nfreqs, Nfreqs)
            data covariance, used for errorbars
        scalar : float, optional
            Power spectrum normalization scalar to multiply into M
        cosmo : hera_pspec Cosmo_Conversions object
            Adopted cosmology. Default is hera_pspec default.
        """
        self.x1 = x1
        if x2 is None:
            self.x2 = x1
        else:
            self.x2 = x2
        self.C = C
        self.freqs = freqs
        self.Nfreqs = len(freqs)
        if scalar is None:
            scalar = 1
        self.scalar = scalar
        self.container = isinstance(x1, DC)
        if cosmo is None:
            self.cosmo = hp.conversions.Cosmo_Conversions()
        else:
            self.cosmo = cosmo
        self.avg_f = np.mean(freqs)
        self.avg_z = self.cosmo.f2z(self.avg_f * 1e6)
        self.t2k = self.cosmo.tau_to_kpara(self.avg_z)
        
    def _check_type(self, A):
        if A is not None:
            if self.container:
                assert isinstance(A, DC)
            else:
                assert isinstance(A, np.ndarray)

    def set_R(self, R):
        """
        Set weighting matrix for QE.
        For proper OQE, this should be C^-1
        
        Parameters
        ----------
        R : ndarray or DataContainer (Nfreqs, Nfreqs)
        
        Results
        -------
        self.R
        """
        self._check_type(R)
        self.R = R
   
    def _compute_uE(self, R, Q):
        if isinstance(R, DC):
            return DC({k: self._compute_uE(R[k], Q) for k in R})
        return 0.5 * np.array([R.T.conj() @ Qa @ R for Qa in Q])

    def _compute_H(self, uE, Q):
        if isinstance(uE, DC):
            return DC({k: self._compute_H(uE[k], Q) for k in uE})
        return np.array([[np.trace(uEa @ Qb) for Qb in Q] for uEa in uE])

    def compute_H(self, Nbps=None, bp_start=0, prior=None, enforce_real=True):
        """
        Compute response matrix given self.R.
        For R = C^-1 this is the Fisher matrix
        
        Parameters
        ----------
        Nbps : int
            Number of bandpowers. Default is Nfreqs.
        bp_start : int
            Index of Q to start DFT matrix.
            Default is 0.
        prior : ndarray or DataContainer (Ndelays,)
            Bandpower prior. Re-defines Q^_a = prior_a * Q_a
            And defines p^_a = p_a / prior_a
        enforce_real : bool
            If True, take real component of H matrix,
            assuming imaginary component is numerical noise.

        Results
        -------
        self.uE, self.H
        """
        if not hasattr(self, 'R'):
            raise ValueError("No R matrix attached to object")
        # compute Q = dC/dp
        if Nbps is None:
            Nbps = self.Nfreqs
        # get DFT vectors, separable components of Q matrix
        qft = np.fft.fftshift([np.fft.ifft(np.eye(Nbps), axis=-1)[i] for i in range(Nbps)], axes=0)

        # create Nbps x Nbps Q matrix
        Q = np.array([_q[None, :].T.conj().dot(_q[None, :]) for _q in qft]) * Nbps**2

        # if Nfreqs > Nbps, embed within a larger Q matrix
        if Nbps < self.Nfreqs:
            _Q = np.zeros((Nbps, self.Nfreqs, self.Nfreqs), dtype=np.complex)
            bp_end = bp_start + Nbps
            _Q[:, bp_start:bp_end, bp_start:bp_end] = Q
            Q = _Q
        self.Q = Q
        self.qft = qft

        # compute bandpower k bins
        self.dlys = np.fft.fftshift(np.fft.fftfreq(Nbps, np.diff(self.freqs)[0])) * 1e3
        self.kp = self.dlys * self.t2k / 1e9
        self.kp_mag = np.abs(self.kp)

        # enact prior
        if prior is not None:
            self.prior = prior
            for i, p in enumerate(prior):
                self.Q[i] *= p
        
        # compute un-normed E and then H
        self.uE = self._compute_uE(self.R, self.Q)
        self.H = self._compute_H(self.uE, self.Q)
        if enforce_real:
            self.H = self.H.real

    def _compute_q(self, x1, x2, uE):
        if isinstance(x1, DC):
            return DC({k: self._compute_q(x1[k], x2[k], uE[k]) for k in x1})
        # this is x1^dagger uE x2, but looks weird due to shape of x1, x2
        return np.array([np.diagonal(x1.conj() @ uEa @ x2.T) for uEa in uE])

    def compute_q(self):
        """
        Compute q: un-normalized band powers
        Must first compute_H
        
        Results
        -------
        self.q
        """
        if not hasattr(self, 'H'):
            raise ValueError("Must first compute_H")
        self.q = self._compute_q(self.x1, self.x2, self.uE)

    def _compute_M(self, norm, H):
        if isinstance(H, DC):
            return DC({k: self._compute_M(norm, H[k]) for k in H})
        if norm == 'I':
            Hsum = np.sum(H, axis=1)
            return np.diag(1. / Hsum) * self.scalar
            #return np.diag(np.true_divide(1.0, Fsum, where=~np.isclose(Fsum, 0, atol=1e-15)))
        elif norm == 'H^-1':
            return np.linalg.inv(H) * self.scalar
        elif norm == 'H^-1/2':
            u,s,v = np.linalg.svd(H)
            M = v.T.conj() @ np.diag(1/np.sqrt(s)) @ u.T.conj()
            W = M @ H
            # normalize
            M /= W.sum(axis=1)[:, None]
            return M * self.scalar
        else:
            raise ValueError("{} not recognized".format(norm))

    def _compute_p(self, M, q):
        if isinstance(M, DC):
            return DC({k: self._compute_p(M[k], q[k]) for k in M})
        return M @ q
    
    def _compute_W(self, M, H):
        if isinstance(M, DC):
            return DC({k: self._compute_W(M[k], H[k]) for k in M})
        return M @ H
    
    def _compute_E(self, M, uE):
        if isinstance(M, DC):
            return DC({k: self._compute_E(M[k], uE[k]) for k in M})
        return np.array([np.sum(m[:, None, None] * uE, axis=0) for m in M])

    def _compute_b(self, C, E):
        if isinstance(C, DC):
            return DC({k: self._compute_b(C[k], E[k]) for k in C})
        return np.array([np.trace(C @ Ea) for Ea in E])
    
    def _compute_V(self, C, E):
        if isinstance(C, DC):
            return DC({k: self._compute_V(C[k], E[k]) for k in C})
        return 2 * np.array([[np.trace(C @ Ea @ C @ Eb) for Eb in E] for Ea in E])

    def compute_p(self, norm='I', C_data=None, C_bias=None, sph_avg=True):
        """
        Compute p: normalized bandpowers
        Must first compute_q
        Also computes window function W, normalized
        E matrix nE, bandpower covariance V, and
        bandpower bias term b.
        
        Parameters
        ----------
        norm : str, ['I', 'H^-1', 'H^-1/2']
            Bandpower normalization matrix type
        C_data : ndarray or DataContainer (Nfreqs, Nfreqs), optional
            Data covariance for errorbar estimation.
            Default is self.C
        C_bias : ndarray or DataContainer (Nfreqs, Nfreqs), optional
            Data covariance for bias term.
            Default is no bias term.
        sph_avg : bool,
            If True, perform spherical average onto k_mag axis
            
        Results
        -------
        self.M, self.W, self.E, self.p, self.V, self.b
        """
        if not hasattr(self, 'q'):
            raise ValueError('Must first compute_q')
        self._check_type(C_data)
        self._check_type(C_bias)
        self.norm = norm
        self.kp_mag = np.abs(self.kp)
        # get normalization matrix
        self.M = self._compute_M(norm, self.H)
        # compute window functions
        self.W = self._compute_W(self.M, self.H) / self.scalar
        # compute normalized bandpowers
        self.p = self._compute_p(self.M, self.q)
        # compute normalized E matrix
        self.E = self._compute_E(self.M, self.uE)
        # compute bandpower covariance
        if C_data is None:
            C_data = self.C
        if C_data is not None:
            self.V = self._compute_V(C_data, self.E)
        # compute bias term
        if C_bias is not None:
            self.b = self._compute_b(C_bias, self.E)[:, None]
        else:
            self.b = np.zeros_like(self.p)
        if sph_avg:
            self.spherical_average()

    def spherical_average(self, kp_mag=None):
        """
        Spherical average onto |k| axis

        Parameters
        ----------
        kp_mag : ndarray of k values. Default is |self.kp|

        p_cyl = A p_sph
        p_sph = [A.T C_cyl^-1 A]^-1 A.T C_cyl^-1 p_cyl
        C_sph = [A.T C_cyl^-1 A]^-1
        W_sph = [A.T C_cyl^-1 A]^-1 A.T C_cyl^-1 W_cyl A
        """
        # identity weighting if no errors
        if not hasattr(self, 'V'):
            self.V = np.eye(self.p.shape[0])

        if kp_mag is None:
            self.kp_mag = np.unique(np.abs(self.kp))
        else:
            self.kp_mag = kp_mag

        # construct A matrix
        A = np.zeros((len(self.kp), len(self.kp_mag)))
        for i, k in enumerate(self.kp):
            A[i, np.argmin(np.abs(self.kp_mag - np.abs(k)))] = 1.0

        # get p_sph
        Vinv = np.linalg.inv(self.V)
        AtVinv = A.T @ Vinv
        AtVinvAinv = np.linalg.inv(AtVinv @ A)
        self.p = AtVinvAinv @ AtVinv @ self.p
        self.b = AtVinvAinv @ AtVinv @ self.b
        
        # get V_sph
        self.V = AtVinvAinv

        # get W_sph
        self.W = AtVinvAinv @ AtVinv @ self.W @ A

    def _compute_dsq(self, p, b, V):
        kfac = self.kp_mag[:, None]**3 / 2 / np.pi**2
        dsq_p = p * kfac
        dsq_b = b * kfac
        if V is not None:
            Ik = np.diag(kfac.squeeze())
            dsq_V = Ik @ V @ Ik
        else:
            dsq_V = None
        return dsq_p, dsq_b, dsq_V

    def compute_dsq(self):
        """
        Compute DelSquare

        Result
        -----
        self.dsq
        self.dsq_b
        self.dsq_V
        """
        self.dsq, self.dsq_b, self.dsq_V = self._compute_dsq(self.p, self.b, self.V)

