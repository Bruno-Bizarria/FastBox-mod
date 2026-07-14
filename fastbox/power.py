import numpy as np
import numpy.fft as fft
import pyccl as ccl

class Power(object):
    def __init__(self, box):
        """
        An object to manage power spectrum estimation.
        
        Parameters:
            box (CosmoBox):
                Object containing a simulation box.
        """
        self.box = box

    def get_window_correction_grid(self, method=None):
        """
        Calculates the 3D squared window function |W(k)|^2 for mass assignment correction.

        It implements equation 18 of https://arxiv.org/abs/astro-ph/0409240
        
        Parameters: 
        -----------
        method : 'CIC' of 'NGP' 
            Method to consider
            
        Returns:
        --------
        W2_k : ndarray
            Window function.
        
        """
        if method is None or method.upper() not in ['NGP', 'CIC']:
            raise ValueError("Valid methods: 'NGP' and 'CIC'")
            
        p = 1 if method.upper() == 'NGP' else 2
        
        # 1. Get the 1D sinc function along one axis
        w1d = np.sinc(np.fft.fftfreq(self.box.N))
        
        # 2. Square the 1D window
        w1d_sq = w1d**2
        
        # 3. Broadcast to 3D axes (X, Y, Z)
        Wx2 = w1d_sq[:, None, None]
        Wy2 = w1d_sq[None, :, None]
        Wz2 = w1d_sq[None, None, :]
        
        # 4. Combine and raise to the assignment scheme power
        W2_k = (Wx2 * Wy2 * Wz2)**p
        
        return W2_k

    def unweighted_power(self, delta_x1, delta_x2=None):
        """
        Calculates the 1D spherically averaged auto or cross power spectrum.
        
        Parameters: 
        -----------
        delta_x1 : ndarray 
            overdensity mesh
        delta_x2 : ndarray
            Second overdensity mesh. Default None. If not None, returns cross-power
        
        Returns:
        --------
        kc     : ndarray
            Center of k bins
        vals   : ndarray
            power spectrum (auto or cross, depending on delta_x2)
        stddev :
            simple calculation of standard deviation of the estimator
        """
        if delta_x1 is None:
            raise ValueError('Need to specify field delta_x1')

        # FFT of the first field
        delta_k1 = fft.fftn(delta_x1)

        # ---------------------------------------------------------
        # AUTO-CORRELATION
        # ---------------------------------------------------------
        if delta_x2 is None:
            print('Calculating auto-correlation power spectrum...')
            pk = delta_k1 * np.conj(delta_k1)
            pk = pk.real / self.box.boxfactor

        # ---------------------------------------------------------
        # CROSS-CORRELATION
        # ---------------------------------------------------------
        else:
            print('Calculating cross-correlation power spectrum')
            delta_k2 = fft.fftn(delta_x2)
            
            # Cross power is the real part of (delta_1 * conj(delta_2))
            pk = delta_k1 * np.conj(delta_k2)
            pk = pk.real / self.box.boxfactor
            

        # ---------------------------------------------------------
        # K-BINNING
        # ---------------------------------------------------------
        L_max = max(self.box.Lx, self.box.Ly, self.box.Lz)
        k_min = 2.0 * np.pi / L_max
        k_bin = k_min  
        
        L_min = min(self.box.Lx, self.box.Ly, self.box.Lz)
        k_nyq = np.pi * self.box.N / L_min 
        
        bins = np.arange(k_min, k_nyq + k_bin, k_bin)
        kc = 0.5 * (bins[1:] + bins[:-1])
        
        vals = np.zeros(kc.size)
        stddev = np.zeros(kc.size)
        
        idxs = np.digitize(self.box.k.flatten(), bins)
        pk_flat = pk.flatten()
        
        for i in range(1, bins.size):
            ii = (idxs == i)
            if np.any(ii):
                vals[i-1] = np.mean(pk_flat[ii])
                stddev[i-1] = np.std(pk_flat[ii]) / np.sqrt(np.sum(ii))
            else:
                vals[i-1] = np.nan
                stddev[i-1] = np.nan
        print('DONE')
        
        return kc, vals, stddev

    def model_obs_power_IM(self, km, pkm, bias, Tb, sigdeg, rsd=True, sigma_nl=120):
        """
        Computes the theoretical observational power spectrum by forward-modeling 
        the theoretical matter power spectrum on the 3D FFT grid.
        Includes Beam smoothing, Channel smoothing, FFT Discretization, and RSD.
        
        Parameters:
        -----------
        km, pkm : array_like
            The 1D theoretical matter power spectrum (wavenumbers and power).
        bias : float
            The linear bias factor (b)
        Tb : float
            The mean brightness temperature (signal amplitude)
        sigdeg : float
            Standard deviation of the Gaussian beam in degrees. Set 0 to ignore beam.
        rsd : Bool
            Set True to apply Kaiser and FoG effects.
        sigma_nl : float
            Velocity dispersion in km/s for Finger of God damping. Set 0 to ignore FoG.
            
        Returns:
        --------
        kc, pk_model, stddev : ndarray
            The binned, forward-modeled 1D power spectrum.
        """
        
        # =======================================================
        # 1. Calculate Physical Scales & Cosmology (R_beam, R_chan, R_nl)
        # =======================================================
        z = self.box.redshift
        scale_factor = 1.0 / (1.0 + z)
        h = self.box.cosmo['h']
        Hz = 100. * h * ccl.h_over_h0(self.box.cosmo, scale_factor)
        
        # Transverse Beam Scale
        if sigdeg > 0:
            chi_mpc = ccl.comoving_radial_distance(self.box.cosmo, scale_factor)
            R_beam = np.radians(sigdeg) * (chi_mpc * h)
        else:
            R_beam = 0.0
        '''    
        # LoS Channel Scale
        if channel:
            freqs = self.box.freq_array()
            dnu_mhz = np.abs(np.mean(np.diff(freqs)))
            nu_21 = 1420.40575177
            c_kms = 299792.458
            R_chan = (c_kms * (1.0 + z)**2) / (Hz * nu_21) * dnu_mhz * h
        else:
            R_chan = 0.0
            
        # RSD Linear Growth Rate (f) and FoG Scale (R_nl)
        if rsd:
            f = ccl.growth_rate(self.box.cosmo, scale_factor)
            if sigma_nl > 0:
                # Convert km/s to comoving Mpc/h
                R_nl = (sigma_nl * (1.0 + z) / Hz) * h
            else:
                R_nl = 0.0
        '''
        # =======================================================
        # 2. Create the exact 3D Fourier Grid
        # =======================================================
        kx = 2.0 * np.pi * np.fft.fftfreq(self.box.N, d=self.box.Lx / self.box.N)
        ky = 2.0 * np.pi * np.fft.fftfreq(self.box.N, d=self.box.Ly / self.box.N)
        kz = 2.0 * np.pi * np.fft.fftfreq(self.box.N, d=self.box.Lz / self.box.N)
        
        kx3d = kx[:, None, None]
        ky3d = ky[None, :, None]
        kz3d = kz[None, None, :]
        
        # k_perp (X, Y) and k_parallel (Z)
        k_perp_sq = kx3d**2 + ky3d**2
        k_par_sq = kz3d**2
        k_mag_sq = k_perp_sq + k_par_sq
        k_mag = np.sqrt(k_mag_sq)
        
        # =======================================================
        # 3. Evaluate 3D Theory & Apply RSD + Damping Factors
        # =======================================================
        # Interpolate the theoretical 1D P(k) onto the 3D grid
        pk_3d = np.interp(k_mag.flatten(), km, pkm).reshape(k_mag.shape)
        
        # --- RSD (Kaiser + FoG) ---
        if rsd:
            # Safely calculate mu^2 = k_parallel^2 / k_mag^2 (avoid division by zero at k=0)
            # FIXED: Used np.zeros_like(k_mag_sq) to match the broadcast shape
            mu_sq = np.divide(k_par_sq, k_mag_sq, out=np.zeros_like(k_mag_sq), where=(k_mag_sq != 0))
            
            # Kaiser Factor
            rsd_factor = (bias + f * mu_sq)**2
            
            # FoG Damping (Gaussian velocity dispersion)
            if sigma_nl > 0:
                D2_fog = np.exp(-k_par_sq * (R_nl**2))
            else:
                D2_fog = 1.0
                
            pk_3d = pk_3d * rsd_factor * D2_fog
        else:
            # No RSD: just use isotropic bias
            pk_3d = pk_3d * (bias**2)


        pk_3d = pk_3d * (Tb**2)
        
        # --- Observational Damping ---
        if sigdeg > 0:
            D2_beam = np.exp(-k_perp_sq * (R_beam**2))
        else:
            D2_beam = 1.0
        '''    
        if channel:
            k_par = np.sqrt(k_par_sq)
            D2_chan = (np.sinc((k_par * R_chan) / (2.0 * np.pi)))**2
        else:
            D2_chan = 1.0
        '''    
        # Combine all effects
        pk_3d_obs = pk_3d * D2_beam# * D2_chan 
        
        # =======================================================
        # 4. Bin the 3D Model into 1D k-shells
        # =======================================================
        L_max = max(self.box.Lx, self.box.Ly, self.box.Lz)
        k_min = 2.0 * np.pi / L_max
        k_bin = k_min  
        
        L_min = min(self.box.Lx, self.box.Ly, self.box.Lz)
        k_nyq = np.pi * self.box.N / L_min 
        
        bins = np.arange(k_min, k_nyq + k_bin, k_bin)
        kc = 0.5 * (bins[1:] + bins[:-1])
        
        vals = np.zeros(kc.size)
        stddev = np.zeros(kc.size)
        
        idxs = np.digitize(k_mag.flatten(), bins)
        pk_flat = pk_3d_obs.flatten()
        
        for i in range(1, bins.size):
            ii = (idxs == i)
            if np.any(ii):
                vals[i-1] = np.mean(pk_flat[ii])
                stddev[i-1] = np.std(pk_flat[ii]) / np.sqrt(np.sum(ii))
            else:
                vals[i-1] = np.nan
                stddev[i-1] = np.nan
        print('DONE')
        
        return kc, vals, stddev

    def model_obs_power_gal(self, km, pkm, bias, MAS='NGP', rsd=True, sigma_nl=120):
        """
        Computes the theoretical observational power spectrum for galaxies
        by forward-modeling the matter power spectrum on the 3D FFT grid.
        Includes Mass Assignment Scheme (MAS) window and RSD.
        
        Parameters:
        -----------
        km, pkm : array_like
            The 1D theoretical matter power spectrum (wavenumbers and power).
        bias : float
            The linear bias factor (b)
        MAS : 'NGP' or 'CIC' or None
            mass assignment method to correct.
        rsd : Bool
            Set True to apply Kaiser and FoG effects.
        sigma_nl : float
            Velocity dispersion in km/s for Finger of God damping. Set 0 to ignore FoG.
            
        Returns:
        --------
        kc, pk_model, stddev : ndarray
            The binned, forward-modeled 1D power spectrum.
        """
        
        # =======================================================
        # 1. Create the exact 3D Fourier Grid
        # =======================================================
        kx = 2.0 * np.pi * np.fft.fftfreq(self.box.N, d=self.box.Lx / self.box.N)
        ky = 2.0 * np.pi * np.fft.fftfreq(self.box.N, d=self.box.Ly / self.box.N)
        kz = 2.0 * np.pi * np.fft.fftfreq(self.box.N, d=self.box.Lz / self.box.N)
        
        kx3d = kx[:, None, None]
        ky3d = ky[None, :, None]
        kz3d = kz[None, None, :]
        
        # k_perp (X, Y) and k_parallel (Z)
        k_perp_sq = kx3d**2 + ky3d**2
        k_par_sq = kz3d**2
        k_mag_sq = k_perp_sq + k_par_sq
        k_mag = np.sqrt(k_mag_sq)

        # =======================================================
        # 2. Evaluate 3D Theory & Apply RSD 
        # =======================================================
        # Interpolate the theoretical 1D P(k) onto the 3D grid
        pk_3d = np.interp(k_mag.flatten(), km, pkm).reshape(k_mag.shape)
        
        # Cosmology for RSD
        z = self.box.redshift
        scale_factor = 1.0 / (1.0 + z)

        if rsd:
            h = self.box.cosmo['h']
            Hz = 100. * h * ccl.h_over_h0(self.box.cosmo, scale_factor)
            f = ccl.growth_rate(self.box.cosmo, scale_factor)
            
            if sigma_nl > 0:
                R_nl = (sigma_nl * (1.0 + z) / Hz) * h
            else:
                R_nl = 0.0
                
            # Safely calculate mu^2
            mu_sq = np.divide(k_par_sq, k_mag_sq, out=np.zeros_like(k_mag_sq), where=(k_mag_sq != 0))
            
            # Kaiser Factor and FoG
            rsd_factor = (bias + f * mu_sq)**2
            D2_fog = np.exp(-k_par_sq * (R_nl**2)) if sigma_nl > 0 else 1.0
            
            pk_3d = pk_3d * rsd_factor * D2_fog
        else:
            # If no RSD, just apply the isotropic isotropic clustering
            pk_3d = pk_3d * (bias**2)

        # =======================================================
        # 3. Apply Discretization Window (MAS)
        # =======================================================
        if MAS is not None:
            # Fixed: Removed self.N argument to match your earlier definition
            W2_k = self.get_window_correction_grid(method=MAS)
            pk_3d = pk_3d * W2_k

        # =======================================================
        # 4. Bin the 3D Model into 1D k-shells
        # =======================================================
        L_max = max(self.box.Lx, self.box.Ly, self.box.Lz)
        k_min = 2.0 * np.pi / L_max
        k_bin = k_min  
        
        L_min = min(self.box.Lx, self.box.Ly, self.box.Lz)
        k_nyq = np.pi * self.box.N / L_min 
        
        bins = np.arange(k_min, k_nyq + k_bin, k_bin)
        kc = 0.5 * (bins[1:] + bins[:-1])
        
        vals = np.zeros(kc.size)
        stddev = np.zeros(kc.size)
        
        idxs = np.digitize(k_mag.flatten(), bins)
        pk_flat = pk_3d.flatten()
        
        for i in range(1, bins.size):
            ii = (idxs == i)
            if np.any(ii):
                vals[i-1] = np.mean(pk_flat[ii])
                stddev[i-1] = np.std(pk_flat[ii]) / np.sqrt(np.sum(ii))
            else:
                vals[i-1] = np.nan
                stddev[i-1] = np.nan
        print('DONE')
        
        return kc, vals, stddev

    def model_obs_power_CC(self, km, pkm, bias_HI, bias_gal, Tb, sigdeg, MAS='NGP', rsd=True, sigma_nl=120):
        """
        Computes the theoretical Cross-Correlation power spectrum by forward-modeling 
        the matter power spectrum on the 3D FFT grid.
        
        Parameters:
        -----------
        km, pkm : array_like
            The 1D theoretical matter power spectrum (wavenumbers and power).
        bias_HI : float
            The linear bias factor of the Intensity Mapping tracer.
        bias_gal : float
            The linear bias factor of the Galaxy tracer.
        Tb : float
            The mean brightness temperature (signal amplitude).
        sigdeg : float
            Standard deviation of the Gaussian beam in degrees.
        MAS : 'NGP' or 'CIC' or None
            Mass assignment method used.
        rsd : Bool
            Set True to apply Kaiser and FoG effects.
        sigma_nl : float
            Velocity dispersion in km/s for Finger of God damping.
        """
        import pyccl as ccl
        import numpy as np
        
        # =======================================================
        # 1. Calculate Physical Scales & Cosmology
        # =======================================================
        z = self.box.redshift
        scale_factor = 1.0 / (1.0 + z)
        h = self.box.cosmo['h']
        Hz = 100. * h * ccl.h_over_h0(self.box.cosmo, scale_factor)
        
        # Transverse Beam Scale
        if sigdeg > 0:
            chi_mpc = ccl.comoving_radial_distance(self.box.cosmo, scale_factor)
            R_beam = np.radians(sigdeg) * (chi_mpc * h)
        else:
            R_beam = 0.0

        # Calculate RSD parameters
        if rsd:
            f = ccl.growth_rate(self.box.cosmo, scale_factor)
            if sigma_nl > 0:
                R_nl = (sigma_nl * (1.0 + z) / Hz) * h
            else:
                R_nl = 0.0

        # =======================================================
        # 2. Create the exact 3D Fourier Grid
        # =======================================================
        kx = 2.0 * np.pi * np.fft.fftfreq(self.box.N, d=self.box.Lx / self.box.N)
        ky = 2.0 * np.pi * np.fft.fftfreq(self.box.N, d=self.box.Ly / self.box.N)
        kz = 2.0 * np.pi * np.fft.fftfreq(self.box.N, d=self.box.Lz / self.box.N)
        
        kx3d = kx[:, None, None]
        ky3d = ky[None, :, None]
        kz3d = kz[None, None, :]
        
        k_perp_sq = kx3d**2 + ky3d**2
        k_par_sq = kz3d**2
        k_mag_sq = k_perp_sq + k_par_sq
        k_mag = np.sqrt(k_mag_sq)
        
        # =======================================================
        # 3. Evaluate 3D Theory & Apply RSD + Damping Factors
        # =======================================================
        pk_3d = np.interp(k_mag.flatten(), km, pkm).reshape(k_mag.shape)
        
        # --- RSD (Kaiser + FoG) ---
        if rsd:
            mu_sq = np.divide(k_par_sq, k_mag_sq, out=np.zeros_like(k_mag_sq), where=(k_mag_sq != 0))
            
            # Cross-Correlation Kaiser Factor: (b_HI + f*mu^2) * (b_gal + f*mu^2)
            rsd_factor = (bias_HI + f * mu_sq) * (bias_gal + f * mu_sq)
            
            D2_fog = np.exp(-k_par_sq * (R_nl**2)) if sigma_nl > 0 else 1.0
                
            pk_3d = pk_3d * rsd_factor * D2_fog
        else:
            # No RSD: just use isotropic cross-bias
            pk_3d = pk_3d * (bias_HI * bias_gal)

        # Apply single Temperature scaling for Cross-Correlation
        pk_3d = pk_3d * Tb
        
        # --- Observational Damping ---
        D2_beam = np.exp(-k_perp_sq * (R_beam**2)) if sigdeg > 0 else 1.0
  
        # Cross-power uses exactly one power of the beam (sqrt(D2_beam) = B_beam)
        pk_3d_obs = pk_3d * np.sqrt(D2_beam)
        
        # Discretization Window
        if MAS is not None:
            # Retain TWO powers of pixelization for Cross-Correlation, applied directly to pk_3d_obs
            W2_k = self.get_window_correction_grid(method=MAS)
            pk_3d_obs = pk_3d_obs * np.sqrt(W2_k)

        # =======================================================
        # 4. Bin the 3D Model into 1D k-shells
        # =======================================================
        L_max = max(self.box.Lx, self.box.Ly, self.box.Lz)
        k_min = 2.0 * np.pi / L_max
        k_bin = k_min  
        
        L_min = min(self.box.Lx, self.box.Ly, self.box.Lz)
        k_nyq = np.pi * self.box.N / L_min 
        
        bins = np.arange(k_min, k_nyq + k_bin, k_bin)
        kc = 0.5 * (bins[1:] + bins[:-1])
        
        vals = np.zeros(kc.size)
        stddev = np.zeros(kc.size)
        
        idxs = np.digitize(k_mag.flatten(), bins)
        pk_flat = pk_3d_obs.flatten()
        
        for i in range(1, bins.size):
            ii = (idxs == i)
            if np.any(ii):
                vals[i-1] = np.mean(pk_flat[ii])
                stddev[i-1] = np.std(pk_flat[ii]) / np.sqrt(np.sum(ii))
            else:
                vals[i-1] = np.nan
                stddev[i-1] = np.nan
        print('DONE')
        
        return kc, vals, stddev 




    
        
    def matter_power_spectrum(self, k, rsd=False, sigma_nl=0):
        """
        Calculate the theoretical nonlinear power spectrum for the given 
        cosmological parameters, using CCL. Does not depend on the realisation.
        
        Parameters:
        -----------
        k : array_like
            k values to evaluate power spectrum
        rsd : Bool
            Set True to consider Kaiser and FoG
        sigma_nl : float
            non-linear velocity. Set 0 to not consider FoG
            
        Returns:
        --------
            k, pk (array_like):
                Wavenumbers, from 10^-3.5 to 10^1, in Mpc^-1, and the 
                theoretical nonlinear power spectrum, in (Mpc)^3.
        """

        
        # 1. Calculate the real-space matter power spectrum P_m(k)
        pk = ccl.nonlin_matter_power(self.box.cosmo, k=k, a=self.box.scale_factor)
        
        if rsd is False:
            return k, pk
            
        elif rsd is True:
            # 2. Cosmology & Redshift parameters
            z = (1.0 / self.box.scale_factor) - 1.0
            
            # Linear growth rate (f)
            f = ccl.growth_rate(self.box.cosmo, self.box.scale_factor)
            
            # Hubble parameter at z in km/s/Mpc
            Hz = 100. * self.box.cosmo['h'] * ccl.h_over_h0(self.box.cosmo, self.box.scale_factor)
            
            # 3. Finger of God (FoG) Scale
            # Since k is in Mpc^-1, R_nl must be in Mpc
            if sigma_nl > 0:
                R_nl = (sigma_nl * (1.0 + z)) / Hz
            else:
                R_nl = 0.0
                
            # 4. Spherically Average over mu (angle to the line of sight)
            n_mu = 200
            mu = np.linspace(0, 1, n_mu)
            
            # Create a 2D grid for k and mu combinations
            k_grid, mu_grid = np.meshgrid(k, mu, indexing='ij')
            
            # Broadcast 1D P_m(k) to the 2D grid shape: (len(k), len(mu))
            pk_grid = pk[:, None]
            
            # Kaiser effect: (b + f*mu^2)^2. For pure matter, bias b = 1.0
            kaiser_factor = (1.0 + f * (mu_grid**2))**2
            
            # FoG Damping: exp(-(k * mu * R_nl)^2)
            if sigma_nl > 0:
                fog_damping = np.exp(-((k_grid * mu_grid * R_nl)**2))
            else:
                fog_damping = 1.0
                
            # Combine to get the 2D Redshift-Space Power Spectrum
            pk_2d = pk_grid * kaiser_factor * fog_damping
            
            # Integrate (average) over mu to get the 1D spherically averaged P(k)
            pk_1d = np.trapz(pk_2d, x=mu, axis=1)
            
            return k, pk_1d
            
        else:
            raise ValueError('RSD option must be either True or False')
        