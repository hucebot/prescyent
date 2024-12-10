# Datasets

We implemented classes for some homebred dataset along with SOTA datasets for the task of human motion prediction to the library.  

Import all of theses datasets and their configs from `prescyent.dataset`.  

## Downloadable Datasets

### AndyDataset

This in-house dataset includes measurements of human movements and forces during the execution of different manual tasks.  
For the HDF5 dataset we only extracted the information from xsens' .mvnx files  
Further descriptions, original data and additional data for this dataset can be found [here](https://zenodo.org/records/3254403#.Y_9fwBeZMVk).  

This hdf5 dataset contains the following features from the xsens motion capture, at each time frame at 240Hz:
- x, y, z coordinates and quaternions for each of the 23 segments
- acceleration
- angular acceleration
- angular velocity
- center of mass
- joint angles
- joint angles XZY
- sensor free acceleration
- velocity

[Config](configuration_files.rst#andydatasetconfig)

### H36MDataset

Subset of the [h36m dataset](http://vision.imar.ro/human3.6m/description.php) used in [numerous benchmarks](https://paperswithcode.com/dataset/human3-6m), such as in Human Pose Forecasting.  
This subsets contains only the exponential maps for all joints of the actors.  
It was originally downloaded from a [stanford backup](http://www.cs.stanford.edu/people/ashesh/h3.6m.zip) (down now)  

The HDF5 file includes all of the trajectories into one single file, following the concepts used in the prescyent library.  
For each traj, we have the coordinates and rotations for the 32 joints, inferred from the expmap and skeleton infos at 50Hz, as in [siMLPe's implementation](https://github.com/dulucas/siMLPe)  

To use it in the [PreScyent library](https://github.com/hucebot/prescyent/), simply download the .hdf5 file and place it in your data folder (described in your DatasetConfig, default is "data/datasets/")  

[Config](configuration_files.rst#h36mdatasetconfig)

### TeleopIcubDataset

In-house dataset obtained from the whole-body teleoperation of the iCub robot performing various bimanual tasks.  
It contains the following features at each time frame at 100Hz:
- x, y, z coordinates for the left hand, right hand and waist
- center of mass of the robot
- all 32 DoF of the robot.    

Further descriptions and the original data for this dataset can be found [here](https://zenodo.org/records/5913573).  

[Config](configuration_files.rst#teleopicubdatasetconfig)  


## Synthetic Datasets

Some datasets are created by the library and do not require to download a file. We refer to them as Synthetic Datasets

### SCCDataset

Synthetic Dataset generating circular 2D trajectories with more or less noise given the configuration parameters.  

[Config](configuration_files.rst#sccdatasetconfig)

### SSTDataset

Synthetic dataset given configurable starting point, velocity and min_max coordinates, generates a smooth set of trajectories and rotation for one point.  

[Config](configuration_files.rst#sstdatasetconfig)
