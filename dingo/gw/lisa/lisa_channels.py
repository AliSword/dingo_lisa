import numpy as np



class LISALowFrequencyInterferometer(object):
    """ 
        Class to create a LISA interferometer object assuming the low frequency approximation.
        
        Static parameters
        -----------------

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
        Parameters
        ----------

        name : str
            Interferometer name, e.g. LISA1, LISA2
        """
        
        self.name = name 
        
    def antenna_response(self, theta_s, phi_s, theta_l, phi_l, t_ref, mode):  
        """
        Returns the LISA1 and LISA2 antenna pattern functions in the Solar System Barycenter
        
        Parameters
        ----------
        theta_s : float
            Source's polar angle. 
        phi_s : float
            Source's azimuthal angle.
        theta_l : float
            Polar angle of the source's angular momentum.
        phi_l : float
            Azimuthal angle of the source's angular momentum.
        t_ref : float
            Reference time.
        mode : str
            Polarization mode, e.g. plus, cross
        """
        
        costheta = 0.5 * np.cos(theta_s) - 0.5 * np.sqrt(3.) * np.sin(theta_s) * np.cos(LISALowFrequencyInterferometer.phi_0 + 
                    LISALowFrequencyInterferometer.omega * t_ref - phi_s)
        
        phi = LISALowFrequencyInterferometer.alpha_0 + LISALowFrequencyInterferometer.omega * t_ref + np.arctan2(np.sqrt(3.) * np.cos(theta_s) + np.sin(theta_s) * np.cos(LISALowFrequencyInterferometer.phi_0 + LISALowFrequencyInterferometer.omega * t_ref - phi_s) , 2 * np.sin(theta_s) * np.sin(LISALowFrequencyInterferometer.phi_0 + LISALowFrequencyInterferometer.omega * t_ref - phi_s))
        
        psiup = 0.5 * np.cos(theta_l) -0.5 * np.sqrt(3.0) * np.sin(theta_l)*np.cos(LISALowFrequencyInterferometer.phi_0 + 
                    LISALowFrequencyInterferometer.omega * t_ref - phi_l) - costheta * (np.cos(theta_l) * np.cos(theta_s) + np.sin(theta_l) * np.sin(theta_s) * np.cos(phi_l-phi_s))
        
        psidown = 0.5 * np.sin(theta_l) * np.sin(theta_s) * np.sin(phi_l-phi_s) - 0.5 * np.sqrt(3.0) * \
                    np.cos(LISALowFrequencyInterferometer.phi_0 + LISALowFrequencyInterferometer.omega * t_ref) * \
                    (np.cos(theta_l) * np.sin(theta_s) * np.sin(phi_s) - np.cos(theta_s)*np.sin(theta_l)*np.sin(phi_l)) - 0.5 * \
                    np.sqrt(3.0) * np.sin(LISALowFrequencyInterferometer.phi_0 + LISALowFrequencyInterferometer.omega * t_ref) * (np.cos(theta_s) * np.sin(theta_l) * np.cos(phi_l) - np.cos(theta_l) * np.sin(theta_s) * np.cos(phi_s))
        
        psi = np.arctan2(psiup, psidown)
        
        if self.name =='LISA1':
            if mode =='plus':
                return np.sqrt(3.0) * 0.5 * 0.5 * (1 + costheta**2) * np.cos(2 * phi) * np.cos(2 * psi) - costheta * \
                        np.sin(2 * phi) * np.sin(2 * psi)
            elif mode == 'cross':
                return np.sqrt(3.0) * 0.5 * 0.5 * (1 + costheta**2) * np.cos(2 * phi) * np.sin(2 * psi) + costheta * \
                        np.sin(2 * phi) * np.cos(2 * psi)
        elif self.name == 'LISA2':
            if mode =='plus':
                return np.sqrt(3.0) * 0.5 * 0.5 * (1 + costheta**2) * np.sin(2 * phi) * np.cos(2 * psi) + costheta * \
                        np.cos(2 * phi) * np.sin(2 * psi)
            elif mode == 'cross':
                return np.sqrt(3.0) * 0.5 * 0.5 * (1 + costheta**2) * np.sin(2 * phi) * np.sin(2 * psi) - costheta * \
                        np.cos(2 * phi) * np.cos(2 * psi)
        

def get_empty_lisa_interferometer(ifo_name): 
    """ 
    Returns an empty LISA interferometer with the given name. 
    Options are for the different low-frequency channels LISA1 and LISA2.
    """
    # can pickle this in the future
    if ifo_name == "LISA1":
        # static parameters for each detector?
        # I think this should have \bar{\phi}_0 and \bar{\alpha}_0
        return LISALowFrequencyInterferometer(
            name="LISA1",
            INSERT_STATIC_PARAMS_HERE=None
        )
    elif ifo_name == "LISA2":
        # static parameters for each detector?
        # I think this should have \bar{\phi}_0 and \bar{\alpha}_0
        return LISALowFrequencyInterferometer(
            name="LISA2",
            INSERT_STATIC_PARAMS_HERE=None
        )
    else:
        raise ValueError("Unknown LISA interferometer name: {}".format(ifo_name))

    pass 
