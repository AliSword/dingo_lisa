import textwrap
import yaml
import argparse
from multiprocessing import Pool
import pandas as pd
import numpy as np
from threadpoolctl import threadpool_limits

from dingo.gw.dataset.waveform_dataset import WaveformDataset
from dingo.gw.prior import build_prior_with_defaults
from dingo.gw.domains import build_domain
from dingo.gw.waveform_generator import WaveformGenerator, generate_waveforms_parallel
from torchvision.transforms import Compose
from dingo.gw.SVD import SVDBasis, ApplySVD


def generate_parameters_and_polarizations(
    waveform_generator, prior, num_samples, num_processes
):
    """
    Generate a dataset of waveforms based on parameters drawn from the prior.

    Parameters
    ----------
    waveform_generator : WaveformGenerator
    prior : Prior
    num_samples : int
    num_processes : int

    Returns
    -------
    pandas DataFrame of parameters
    dictionary of numpy arrays corresponding to waveform polarizations
    """
    print("Generating dataset of size " + str(num_samples))
    parameters = pd.DataFrame(prior.sample(num_samples))

    if num_processes > 1:
        with threadpool_limits(limits=1, user_api="blas"):
            with Pool(processes=num_processes) as pool:
                polarizations = generate_waveforms_parallel(
                    waveform_generator, parameters, pool
                )
    else:
        polarizations = generate_waveforms_parallel(waveform_generator, parameters)

    # Find cases where waveform generation failed and only return data for successful ones
    wf_failed = np.any(np.isnan(polarizations['h_plus']), axis=1)
    if wf_failed.any():
        idx_failed = np.where(wf_failed)[0]
        idx_ok = np.where(~wf_failed)[0]
        polarizations_ok = {k: v[idx_ok] for k, v in polarizations.items()}
        parameters_ok = parameters.iloc[idx_ok]
        failed_percent = 100 * len(idx_failed) / len(parameters)
        print(f'{len(idx_failed)} out of {len(parameters)} configuration ({failed_percent:.1f}%) failed to generate.')
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(parameters.iloc[idx_failed])
        print(f'Only returning the {len(idx_ok)} successfully generated configurations.')
        return parameters_ok, polarizations_ok

    return parameters, polarizations


def generate_dataset(settings, num_processes):
    """
    Generate a waveform dataset.

    Parameters
    ----------
    settings : dict
        Dictionary of settings to configure the dataset
    num_processes : int

    Returns
    -------
    A dictionary consisting of a parameters DataFrame and a polarizations dictionary of
    numpy arrays.
    """

    prior = build_prior_with_defaults(settings["intrinsic_prior"])
    domain = build_domain(settings["domain"])
    waveform_generator = WaveformGenerator(
        settings["waveform_generator"]["approximant"],
        domain,
        settings["waveform_generator"]["f_ref"],
    )

    dataset_dict = {"settings": settings}

    if "compression" in settings:
        compression_transforms = []

        if "svd" in settings["compression"]:
            svd_settings = settings["compression"]["svd"]

            # Load an SVD basis from file, if specified.
            if "file" in svd_settings:
                print("Loading SVD basis from " + svd_settings["file"])
                basis = SVDBasis()
                basis.from_file(svd_settings["file"])

            # Otherwise, generate the basis based on simulated waveforms.
            else:

                parameters, polarizations = generate_parameters_and_polarizations(
                    waveform_generator,
                    prior,
                    svd_settings["num_training_samples"],
                    num_processes,
                )
                train_data = np.vstack(list(polarizations.values()))
                print("Building SVD basis.")
                basis = SVDBasis()
                basis.generate_basis(train_data, svd_settings["size"])

                del parameters, polarizations, train_data

            compression_transforms.append(ApplySVD(basis))
            dataset_dict["svd_V"] = basis.V

        waveform_generator.transform = Compose(compression_transforms)

    # Generate main dataset
    parameters, polarizations = generate_parameters_and_polarizations(
        waveform_generator, prior, settings["num_samples"], num_processes
    )
    dataset_dict["parameters"] = parameters
    dataset_dict["polarizations"] = polarizations

    dataset = WaveformDataset(dictionary=dataset_dict)
    return dataset


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(
            """\
        Generate a waveform dataset based on a settings file.
        """
        ),
    )
    parser.add_argument(
        "--settings_file",
        type=str,
        required=True,
        help="YAML file containing database settings",
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=1,
        help="Number of processes to use in pool for parallel waveform generation",
    )
    parser.add_argument(
        "--out_file",
        type=str,
        default="waveform_dataset.hdf5",
        help="Name of file for storing dataset.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load settings
    with open(args.settings_file, "r") as f:
        settings = yaml.safe_load(f)

    dataset = generate_dataset(settings, args.num_processes)
    dataset.to_file(args.out_file)


if __name__ == "__main__":
    main()
