"""Script to transform TeleopIcub dataset's csv files to one hdf5 file in the prescyent format"""

from argparse import ArgumentParser

from prescyent.dataset.datasets.teleop_icub import TeleopIcubDataset


DEFAULT_DATA_PATH = "data/datasets/AndyData-lab-prescientTeleopICub"
DEFAULT_GLOB_PATERN = "*.csv"
DEFAULT_HDF5_PATH = "data/datasets/AndyData-lab-prescientTeleopICub.hdf5"


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--data_path", default=DEFAULT_DATA_PATH, help="path to the data directory"
    )
    parser.add_argument(
        "--glob_patern",
        default=DEFAULT_GLOB_PATERN,
        help="pattern used to retreive the list of files to parse in the directory",
    )
    parser.add_argument(
        "--hdf5_path", default=DEFAULT_HDF5_PATH, help="filepath to the created hdf5"
    )

    args = parser.parse_args()
    hdf5_path = args.hdf5_path
    data_path = args.data_path
    glob_patern = args.glob_patern

    TeleopIcubDataset.create_hdf5(hdf5_path, data_path, glob_patern)
