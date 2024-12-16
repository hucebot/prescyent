# Main concepts and classes

The library is split in 4 main modules:  

- **dataset**: where you will find classes to represent the trajectories, their features, and load them as a dataset.  
- **scaler**: where you will find classes to configure and perform scaling on your data.  
- **predictor**: where you will find methods and models to run predictions over trajectories.  
- **evaluator**: where you will find plots, 3d visualizations and evaluation runners.  

And some methods or tools like the logger or the enums are under the utils module  

## Configs
The main classes' attributes are passed through an instance of their _config class. We use pydantic.BaseModel to generate configurations classes with default value and type checking, to ensure that the object's configurable attributes are valid.  
There are configs for datasets, scalers, predictors (they must include the scaler's and dataset's config), and for the training.  
You can see a list of the config classes, their attributes and defaults values here:  
[configuration files](configuration_files.rst).  

## Dataset module

### Trajectory

We call Trajectory a sequence over time, over which we track one or many features, for one or many points.  

A single Trajectory is represented by a tensor, and described by attributes such as their frequency, title, list of name for each of their points, or list of features represented in its tensors. Note that each feature can be represented by 1-N dimensions, for example the feature Coordinates, can be represented in 3D, corresponding to the x, y, z coordinates.  

Our Trajectory tensors have the shape (S, P, D), with the following:  
 - S: number of time frames in the Trajectory  
 - P: number of points we are tracking  
 - D: number of dimensions of the features  

Also a Trajectory may have some additional context, such as images, eye tracking, intent or labels...  
To represent theses, we provide a dictionary of tensors alongside the trajectory_tensor. The constraint over theses context tensors is that we expect them to be at the same frequency and size as the trajectory_tensor.  

Example:  

We have a trajectory tracking a 100Hz the 3D pose of the right hand and left hand of a human performing bimanual manipulation task during 30 seconds. 3D poses are encoded as x,y,z coordinates and qx, qy, qz, qw forming a quaternion to represent the rotation.  

The tensor representing this Trajectory would have the shapes `(2500, 2, 7)`  

Let's say that alongside of the hand pose, we also have the center of mass of the human but at 200Hz. I'll need to subsample my data to 100Hz and align it with my trajectory infos. When it is done, the center_of_mass_tensor is a tensor of floats with the shapes `(2500, 3)`  

Now we can init a Trajectory object with all the data and metadata at my disposal:  

```python
    example_traj = Trajectory(
        tensor=example_tensor,
        frequency=100,
        tensor_features=[CoordinatesXYZ(range(3)), RotationQuat(range(3, 7))],
        context= {"center_of_mass": center_of_mass_tensor}
        file_path="example/path/data",
        title="my_example_traj",
        point_parents=None,
        point_names=["left_hand", "right_hand"]
    )
```

Please note that not all fields are mandatory. You must define your tensor and describe them with its features and frequency.  
Context is optional, and other attributes (file_path, title, point_parents, point_names) are metadata used to plot results and render trajectories and all have default values.  

### TrajectoryHDF5

As we expect to handle huge datasets, this child class of Trajectory has its `.tensor` and `.context` attributes that are loaded and saved to disk on an HDF5 file.  

### Trajectories

This is a set of Trajectory, in a split of train, test and val, with helper functions to access theses and their attributes.  
Trajectories can be loaded from a HDF5 file following the prescyent's format.  

#### HDF5 file description

The HDF5 files that are used in the library to generate the Trajectories as the following format:  
- One file for the whole set of trajectories  
- `frequency`, `point_names` and `point_parents` metadata as root attributes  
- `tensor_feat` group at root with:  
  - a dataset for each feature with its ids, and with `distance_unit` and `feature_class` attributes  
- train, test, val groups at root with:  
  - any amount of subgroup, to keep the original files structure, used to generate the `Trajectory.title`  
  - `traj` dataset with the `Trajectory.tensor`  
  - any amount of `xxx` datasets, that will be put as context as `Trajectory.context[xxx]`  

Note that some datasets used in the library don't have the train, test, val split, it is generated from the config instead of being fixed in the original hdf5 file.  

### Features

Features is a List of Feature describing the trajectory tensor.  
A feature, like `RotationQuat(range(4))` details that the trajectory tensor's values at dims [0, 1, 2, 3] corresponds to a the qw, qx, qy, qz of a quaternion.  
This info is used to:  
- Perform feature conversion when possible: RotationQuat => RotationEuler  
- Have some feature aware distance: RotationQuat.get_distance returns the geodesic distance  
- Compute feature wise scaling in the Scaler module  
- Derivate on the feature for outputs or inputs  

We currently support 4 formats of 3D rotations 'euler, quaternions, rotation_matrices, 6d_representations' and 1D to 3D coordinates.  
Feel free to add support for new features and to define your own get_distance methods.  
If you don't want to bother too much about theses feature wise operations, we also have a "Any" type feature (for which the distance is also Euclidean distance).  

### TrajectoriesDataset

Our base dataset class, which follows the logic of [LightningDatamodule](https://lightning.ai/docs/pytorch/stable/data/datamodule.html).  

The two main functions are:
- `prepare_data`: must be implemented by actual dataset classes to create the Trajectories `.trajectories` attribute  
`setup`: common to all TrajectoriesDataset, with the behavior to generate datasamples and dataloaders based on the `.trajectories` attribute and the DatasetConfig's `.config`  

Theses methods are not called at the initialization of the class. Instead they are created by Lightning following the default behavior used by the `lightning.Trainer`, as it handles the multi processes defined in your `TrainingConfig.accelerator` and `TrainingConfig.devices`. Or they run on the go if you call for `.trajectories` or `.test_datasample` while they weren't instantiated.  

Note that when loading the HDF5 trajectories from one of our dataset classes, to not alter the original HDF5 file, we always create a new tmp hdf5 file in .prepare_data child methods of our datasets.  

### TrajectoryDataSample

Classes to generate the batches using for training and testing.  
Each datasample produces (sample, context).  
Each dataloader produces batched (sample, context, truth).  

If the option `save_on_disk` is True in the `TrajectoriesDataset.config`, then we'll take more disk space and time at initialization to save all generated possible samples on disk, allowing to batch the calls to disk, while the original behavior generates the samples on the go from the Trajectory or TrajectoryHDF5 tensors, taking less space but can be more expensive at runtime, especially TrajectoryHDF5 that makes a lot of calls to disk.  

## Scaler module

This module contains the main Scalers class that initializes, trains and calls the configured scaling methods. 
Depending on the configuration you'll run 0 or n instance of a Scaler from the Scalers class.  
We have **Normalization** and **Standardization** as scaling methods.  
Scalers main class and Scaler instances all have their own train method that is called along the predictor's `.train` method giving a dataloader over the train_trajectories.  
The Scalers method allow to have a feature-wise scaling, instantiating and training scaling methods for given features on given dimensions (note that we are always scaling over the batch and temporal dimensions + the one chosen in the config).  
`scale` and `unscale` methods are called in `Predictor.predict` as a decorator of the function, and inside the `LightningModule.predict_torch` method that is called at each train_ test_ and val_ epochs during Lightning's functions, so it is seamless for most high end usages.  
Just as the predictor class, Scalers can save and load their trained tensors from disk using pickle. Theses methods are also included into the Predictor's load and save methods.  

## Predictor module

### BasePredictor

This is the abstract class defining the API to of the main methods and implementing some of the core functionalities common to each predictor instances.  
For example BasePredictor handles the logging and versioning of newly created predictors, as well as the base behavior for `run` method iterating over a whole trajectory.  
Each predictor must inherit from this class.

### LightningPredictor

This class defines base behavior for all machine learning predictors, instantiating the `LightningModule` and `lightning.Trainer`, with methods to `train` and `test` them.  
It defines also function to save and load models.  

### LightningModule

The LightningModule is a common wrapper around the TorchModule.  
Here we define which loss function will be used to train the model based on the PredictorConfig, and also we the functions used to establish the behavior of the module in the lightning Trainer such as `training_step` or `on_test_epoch_start`. please refer to the [lightning documentation](https://lightning.ai/docs/pytorch/stable/common/lightning_module.html) for further explanations.

### TorchModule

You may want all of your custom `torch.nn.Module` to inherit from this **TorchModule** as it comes with a decorator for the forward method to enables the features `deriv_on_last_frame`, `deriv_output` and `dropout_value`.

### LossFunctions

We implemented here some custom loss functions, such as **MeanTotalDistanceLoss** (and its variations), making use of our Feature objects, calculating the mean feature.distance over each predicted time frame of each predicted point.  
Feel free to add your own loss functions here also and make it available to the `PredictorConfig` through a new valid option in the enum's in `prescyent/utils/enums/loss_functions.py and to the predictors in the `CRITERION_MAPPING` in `prescyent/predictor/lightning/module.py`.  


## Evaluator module

In this module you'll find plots that we use to evaluate our models, such as `plot_mpjpe` or its multi predictor variant `plot_mpjpes` that is called after a module is trained in "train_from_config.py".  
We also implemented a 3d rendering of trajectories that have Coordinates and optionally Rotation as Features (Such as Andy or H36M datasets).  
Finally we created some "runners" that show some logic to run a predictor over a whole trajectory and the ways to feed the input to the predictors.

## AutoPredictor and AutoDataset
Theses Auto_ classes allow to create or load the correct instance of a dataset based on a config file or data using a map to all our predictors and datasets. They return the right instance and load the corresponding config from the Auto_ class' method.  

## Logger
We have a dedicated logger instance in utils.logger. To add some logs or modify the logger, you should import it using `from prescyent.utils.logger import logger`.  
Multiple logger are instantiated inside the library, one for each module, allowing you to increase/decrease the log level on some specific parts of the library, given some LOG_GROUPS.  
To control the log level of a specific logger in your scripts, access the log group using `logger.getChild(LOG_GROUP)` and apply `set_level` or any other method to this child instance of the logger.  