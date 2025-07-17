import numpy as np


class LISAInterferometerList(list):
    """
    Create a list of LISA interferometer objects.
    """

    def __init__(self, interferometers):
        """
        Instantiate a new LISAInterferometerList object.

        Attributes
        ----------

        interferometers: iterable.
        """

        super(LISAInterferometerList, self).__init__()
        if isinstance(interferometers, str):
            raise TypeError("Input must be a list")
        for ifo in interferometers:
            if isinstance(ifo, str):
                ifo = get_empty_lisa_interferometer(ifo)
            self.append(ifo)


class LISALowFrequencyInterferometer(object):
    """ 
    Class to create a LISA interferometer object assuming the low frequency approximation. 
        
    Attributes
    ----------

    phi_0 : float
        Initial orbital phase of the detector.
    alpha_0 : float
        Initial rotational phase of the detector.
    T : float
        Orbital period of the detector.
    omega : float
        Orbital frequency of the detector.
    """

    phi_0 = 0.0
    alpha_0 = 0.0 
    T = np.pi*1e7
    omega = 2.*np.pi/T
    
    def __init__(self, name):
        """ 
        Attributes
        ----------

        name : str
            Interferometer name, e.g. LISA1, LISA2
        """
        
        self.name = name 

    @staticmethod
    def GetEclipticAngularMomentum(iota, theta_s, phi_s, psi):
        """
        Calculate direction of the binary angular momentum in the Solar System Barycenter.

        Parameters
        ----------
        iota: float
            Inclination of the source in radians.
        theta_s: float
            Ecliptic latitude of the source in radians.
        phi_s: float
            Ecliptic longitude of the source in radians.
        psi: float
            Polarization of the source in radians.

        Returns
        -------
        float: The ecliptic latitude and longitude of the angular momentum.
        """

        si = np.sin(iota)
        ci = np.cos(iota)
        cps = np.cos(2*psi) #we are considering that the GW is symmetrical in psi (psi -> psi + pi).
        sps = np.sin(2*psi)
        sths = np.sin(theta_s)
        cths = np.cos(theta_s)
        sphs = np.sin(phi_s)
        cphs = np.cos(phi_s)
        norm = 1/(np.sqrt(sths**2 * sphs**2 + cths**2))
    
        cthl = norm*(si * cps * sths * sphs - si * sps * cths * sths * cphs) + ci * cths
        theta_l = np.arccos(cthl)

        phi_l = np.arctan2(norm * (-si * cps * cths - si * sps * sths**2 * cphs * sphs) + ci * sths * sphs,
                           norm * (si * sps * (sths**2 * sphs**2 + cths**2)) + ci * sths * cphs)
        phi_l = np.mod(phi_l, 2*np.pi)
        return theta_l, phi_l


    def antenna_response(self, theta_s, phi_s, theta_l, phi_l, t_ref, mode):  
        """
        Calculate the response function in the Solar System Barycenter for a given sky location. 
        The computation is based on arXiv:gr-qc/9703068. We assume aligned spin components. 
        
        Parameters
        ----------
        theta_s : float
            Ecliptic latitude of the source in radians. 
        phi_s : float
            Ecliptic longitude of the source in radians.
        theta_l : float
            Ecliptic latitude of the angular momentum in radians.
        phi_l : float
            Ecliptic longitude of the angular momentum in radians.
        psi : float
            Polarization angle of the source in radians.
        t_ref : float
            Reference time.
        mode : str
            Polarization mode, e.g. plus, cross

        Returns
        -------
        float: The antenna response for the specified mode and time/location
        """
        
        cths = np.cos(theta_s)
        sths = np.sin(theta_s)
        sphs = np.sin(phi_s)
        cphs = np.cos(phi_s)
        sthl = np.sin(theta_l)
        cthl = np.cos(theta_l)
        cphl = np.cos(phi_l)
        sphl = np.sin(phi_l)
        orbphsdet = LISALowFrequencyInterferometer.phi_0
        orbfreqdet = LISALowFrequencyInterferometer.omega
        rotphsdet = LISALowFrequencyInterferometer.alpha_0

        costheta = 0.5 * cths - 0.5 * np.sqrt(3.) * sths * np.cos(orbphsdet + orbfreqdet * t_ref - phi_s)
        
        phi = rotphsdet + orbfreqdet * t_ref + np.arctan2(np.sqrt(3.) * cths + sths * np.cos(orbphsdet + orbfreqdet * t_ref - phi_s), 
                                                          2 * sths * np.sin(orbphsdet + orbfreqdet * t_ref - phi_s))
        
        psiup = 0.5 * cthl - 0.5 * np.sqrt(3.0) * sthl * np.cos(orbphsdet + orbfreqdet * t_ref - phi_l) - costheta \
        * (cthl * cths + sthl * sths * np.cos(phi_l - phi_s))
        
        psidown = 0.5 * sthl * sths * np.sin(phi_l - phi_s) - 0.5 * np.sqrt(3.0) * np.cos(orbphsdet + orbfreqdet * t_ref) \
        * (cthl * sths * sphs - cths * sthl * sphl) - 0.5 * np.sqrt(3.0) * np.sin(orbphsdet + orbfreqdet * t_ref)\
        * (cths * sthl * cphl - cthl * sths * cphs)
        
        psi = np.arctan2(psiup, psidown)
        
        if self.name =='LISA1':
            if mode =='plus':
                return np.sqrt(3.0) * 0.5 * (0.5 * (1 + costheta**2) * np.cos(2 * phi) * np.cos(2 * psi) - costheta \
                        * np.sin(2 * phi) * np.sin(2 * psi))
            elif mode == 'cross':
                return np.sqrt(3.0) * 0.5 * (0.5 * (1 + costheta**2) * np.cos(2 * phi) * np.sin(2 * psi) + costheta \
                        * np.sin(2 * phi) * np.cos(2 * psi))
        elif self.name == 'LISA2':
            if mode =='plus':
                return np.sqrt(3.0) * 0.5 * (0.5 * (1 + costheta**2) * np.sin(2 * phi) * np.cos(2 * psi) + costheta \
                        * np.cos(2 * phi) * np.sin(2 * psi))
            elif mode == 'cross':
                return np.sqrt(3.0) * 0.5 * (0.5 * (1 + costheta**2) * np.sin(2 * phi) * np.sin(2 * psi) - costheta \
                        * np.cos(2 * phi) * np.cos(2 * psi))
        

def get_empty_lisa_interferometer(ifo_name): 
    """ 
    Returns an empty LISA interferometer with the given name. 
    Options are for the different low-frequency channels LISA1 and LISA2.

    Parameters
    ----------
    ifo_name: str
        Interferometer name, e.g. LISA1, LISA2
    """
    
    if ifo_name == "LISA1":
        return LISALowFrequencyInterferometer(
            name="LISA1",
        )
    elif ifo_name == "LISA2":
        return LISALowFrequencyInterferometer(
            name="LISA2",
        )
    else:
        raise ValueError("Unknown LISA interferometer name: {}".format(ifo_name))
    pass 
