"""Script to transform AndyDataset dataset's xsens .mvnx files to one hdf5 file in the prescyent format"""

from argparse import ArgumentParser

from prescyent.dataset import AndyDataset


DEFAULT_DATA_PATH = "data/datasets/AndyData-lab-onePerson/xens_mnvx"
DEFAULT_GLOB_PATTERN = "*.mvnx"
DEFAULT_HDF5_PATH = "data/datasets/AndyData-lab-onePerson.hdf5"


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--data_path", default=DEFAULT_DATA_PATH, help="path to the data directory"
    )
    parser.add_argument(
        "--glob_pattern",
        default=DEFAULT_GLOB_PATTERN,
        help="pattern used to retrieve the list of files to parse in the directory",
    )
    parser.add_argument(
        "--hdf5_path", default=DEFAULT_HDF5_PATH, help="filepath to the created hdf5"
    )

    args = parser.parse_args()
    hdf5_path = args.hdf5_path
    data_path = args.data_path
    glob_pattern = args.glob_pattern

    AndyDataset.create_hdf5(hdf5_path, data_path, glob_pattern)
