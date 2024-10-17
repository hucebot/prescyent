# Main concepts and classes

The library is splitted in 3 main modules:

- dataset: where you will find classes to represent the trajectories, their features, and load them as a dataset.
- predictor: where you will find methods and models to run predictions over trajectories.
- evaluator: where you will find plots, 3d visualizations and evaluation runners.

And some methods or tools like the logger or the enums are under the utils module

## Configs

## Logger

## Dataset module

### Trajectory

We call Trajectory a sequence over time, over which we track one or many features, for one or many points.

They are represented by a tensor, and described by attributes such as their frequency, title, list of name for each of their points, or list of features represented in its tensors.

Our Trajectory tensors have the shape (Sequence_len, Number_of_points, Number_of_features).

Example:

We have a trajectory tracking a 100Hz the 3D pose of the right hand and left hand of a human performing bimanual manipulation task during 30 seconds. 3D poses are encoded as x,y,z coordinates and qx, qy, qz, qw forming a quaternion to represent the rotation.

The tensor representing this Trajectory would have the shapes:

`(2500, 2, 7)`
And we would expect this kind of values to init a Trajectory object:

```python
    example_traj = Trajectory(
        tensor=example_tensor,
        frequency=100,
        tensor_features=[CoordinatesXYZ(range(3)), RotationQuat(range(3, 7))],
        file_path="example/path/data",
        title="my_example_traj",
        point_parents=None,
        point_names=["left_hand", "right_hand"]
    )
```

//TODO Explain the default values

### Features

get_distance method

Example:

It would have the tensor_features: `[CoordinatesXYZ([0, 1, 2]), RotationQuat([3, 4, 5, 6])]`

### TrajectoriesDataset

### TrajectoriesDatasetConfig

### MotionDataSample

## Predictor module

### Predictor

### LightningPredictor

### TorchModule

### Extend the Predictor module

### Evaluator module

### AutoPredictor and AutoDataset