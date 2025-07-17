import numpy as np
from scipy.signal.windows import tukey
from scipy.interpolate import interp1d
from bilby.gw.detector import PowerSpectralDensity

#from dingo.gw.prior import default_extrinsic_dict
from dingo.gw.prior import BBHExtrinsicPriorDict


def get_window(window_kwargs):
    """Compute window from window_kwargs."""
    type = window_kwargs["type"]
    if type == "tukey":
        roll_off, T, f_s = (
            window_kwargs["roll_off"],
            window_kwargs["T"],
            window_kwargs["f_s"],
        )
        alpha = 2 * roll_off / T
        w = tukey(int(T * f_s), alpha)
        return w
    else:
        raise NotImplementedError(f"Unknown window type {type}.")


def get_window_factor(window):
    """Compute window factor. If window is not provided as array or tensor but as
    window_kwargs, first build the window."""
    if type(window) == dict:
        window = get_window(window)
    if window is None:
        return 1.0
    return np.sum(window**2) / len(window)


def get_extrinsic_prior_dict(extrinsic_prior, default_dict):
    """Build dict for extrinsic prior by starting with
    default_extrinsic_dict, and overwriting every element for which
    extrinsic_prior is not default.
    TODO: Move to dingo.gw.prior.py?"""
    extrinsic_prior_dict = default_dict.copy()
    for k, v in extrinsic_prior.items():
        if v.lower() != "default":
            extrinsic_prior_dict[k] = v
    return extrinsic_prior_dict


def get_mismatch(a, b, domain, asd_file=None):
    """
    Mistmatch is 1 - overlap, where overlap is defined by
    inner(a, b) / sqrt(inner(a, a) * inner(b, b)).
    See e.g. Eq. (44) in https://arxiv.org/pdf/1106.1021.pdf.

    Parameters
    ----------
    a
    b
    domain
    asd_file

    Returns
    -------

    """
    if asd_file is not None:
        # whiten a and b, such that we can use flat-spectrum inner products below
        psd = PowerSpectralDensity(asd_file=asd_file)
        asd_interp = interp1d(
            psd.frequency_array, psd.asd_array, bounds_error=False, fill_value=np.inf
        )
        asd_array = asd_interp(domain.sample_frequencies)
        a = a / asd_array
        b = b / asd_array
    min_idx = domain.min_idx
    inner_ab = np.sum((a.conj() * b)[..., min_idx:], axis=-1).real
    inner_aa = np.sum((a.conj() * a)[..., min_idx:], axis=-1).real
    inner_bb = np.sum((b.conj() * b)[..., min_idx:], axis=-1).real
    overlap = inner_ab / np.sqrt(inner_aa * inner_bb)
    return 1 - overlap


def get_standardization_dict(
    extrinsic_prior_dict, wfd, selected_parameters, transform=None
):
    """
    Calculates the mean and standard deviation of parameters. This is needed for
    standardizing neural-network input and output.

    Parameters
    ----------
    extrinsic_prior_dict : dict
    wfd : WaveformDataset
    selected_parameters : list[str]
        List of parameters for which to estimate standardization factors.
    transform : Transform
        Operator that will generate samples for parameters contained in
        selected_parameters that are not contained in the intrinsic or extrinsic prior.
        (E.g., H1_time, L1_time_proxy)

    Returns
    -------

    """
    # The intrinsic standardization is estimated based on the entire dataset.
    mean_intrinsic, std_intrinsic = wfd.parameter_mean_std()

    # Some of the extrinsic prior parameters have analytic means and standard
    # deviations. If possible, this will either get these, or else it will estimate
    # them numerically.
    ext_prior = BBHExtrinsicPriorDict(extrinsic_prior_dict)
    mean_extrinsic, std_extrinsic = ext_prior.mean_std(ext_prior.keys())

    # Check that overlap between intrinsic and extrinsic parameters is only
    # due to fiducial values (-> std 0)
    for k in std_intrinsic.keys() & std_extrinsic.keys():
        assert std_intrinsic[k] == 0

    # Merge dicts, overwriting fiducial values for parameters (e.g.,
    # luminosity_distance) in intrinsic parameters by the extrinsic ones
    mean = {**mean_intrinsic, **mean_extrinsic}
    std = {**std_intrinsic, **std_extrinsic}

    # For all remaining parameters that require standardization, we use the transform
    # to sample these and estimate the mean and standard deviation numerically.
    additional_parameters = [p for p in selected_parameters if p not in mean]
    if additional_parameters:
        num_samples = min(100_000, len(wfd.parameters))
        samples = {p: np.empty(num_samples) for p in additional_parameters}
        for n in range(num_samples):
            sample = {"parameters": wfd.parameters.iloc[n].to_dict()}
            sample = transform(sample)
            for p in additional_parameters:
                # This assumes all of the additional parameters are contained within
                # extrinsic_parameters. We have set it up so this is the case for the
                # GNPE proxies and the detector coalescence times.
                samples[p][n] = sample["extrinsic_parameters"][p]
        mean_additional = {p: np.mean(samples[p]).item() for p in additional_parameters}
        std_additional = {p: np.std(samples[p]).item() for p in additional_parameters}

        mean.update(mean_additional)
        std.update(std_additional)

    standardization_dict = {
        "mean": {k: mean[k] for k in selected_parameters},
        "std": {k: std[k] for k in selected_parameters},
    }
    return standardization_dict


def get_signal_duration(chirp_mass, f_min):
    """
    Calculates the time to coalescence [sec] of an inspiraling compact binary.

    Parameters
    ----------
    chirp_mass: float
    f_min: float
    """
    time = 2.18 * (1.21 / chirp_mass)**(5/3) * (100/f_min)**(8/3)
    return time



def get_inner_product(a_f, b_f, Sn_f, delta_f):
    """
    Compute the noise-weighted inner product (a | b)
    
    Parameters:
    - a_f, b_f: arrays of complex frequency-domain data
    - Sn_f: 1D array of one-sided noise PSD evaluated at the same frequencies
    - delta_f: frequency resolution (assume uniform grid)

    Returns:
    - scalar inner product (a|b)
    """
    integrand = np.conj(a_f) * b_f / Sn_f
    return 4 * np.real(np.sum(integrand) * delta_f)


def get_optimal_snr(signal, asd, delta_f, channel=None):
    """
    Compute the optimal SNR for a given channel using the signal and PSD.

    Parameters
    ----------
    signal : dict
        Dictionary with a "waveform" key containing the channels (e.g., LISA1, L1, etc.).
    psd : array
        One-sided noise Power Spectral Density (PSD) evaluated at the same frequencies as the waveform.
    delta_f : float
        Frequency resolution (Δf).
    channel : str or None
        Name of the channel to use. If None, the function selects one automatically.

    Returns
    -------
    snr_opt : float
        Optimal SNR computed as sqrt((h|h)) for the selected channel.
    """

    waveform_dict = signal["waveform"]

    # If no channel specified, choose the first one available
    if channel is None:
        if not waveform_dict:
            raise ValueError("No channel available in signal['waveform']")
        channel = next(iter(waveform_dict))

    if channel not in waveform_dict:
        raise ValueError(f"Channel '{channel}' not found in signal['waveform']")

    if channel not in asd:
        raise ValueError(f"Channel '{channel}' not found in PSD dictionary")

    h_f = waveform_dict[channel]
    asd_f = asd[channel]
    psd_f = asd_f**2  

    # compute SNR^2 = (h|h)
    snr_squared = get_inner_product(h_f, h_f, psd_f, delta_f)

    # compute SNR below a given frequency
    '''if f_max is not None:
        if freqs is None:
            raise ValueError("freqs must be provided if f_max is set")
        mask = freqs <= f_max
        h_f_cut = h_f[mask]
        psd_f_cut = psd_f[mask]
        snr_squared_partial = get_inner_product(h_f_cut, h_f_cut, psd_f_cut, delta_f)
        snr_fraction = snr_squared_partial / snr_squared'''

    return np.sqrt(snr_squared)
