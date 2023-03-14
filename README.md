# PreScyent
Data-driven trajectory prediction library (python)  

# Installation
Clone repository then run this from folder  
For dev environment use  
```bash
pip install -e .
```
Otherwise simply use  
```bash
pip install .
```

# Get Started

PreScyent is a trajectory forecasting library, built upon pytorch_lightning  
It comes with datasets such as [AndyData-lab-onePersonTeleoperatingICub](https://zenodo.org/record/5913573)  
[Human3.6M](http://vision.imar.ro/human3.6m/description.php)
And methods to perform trajectory prediction on this kind of datasets  
  
## Datasets
We call "Trajectory" a sequence in time, of n points of n dimensions.  
It is represented with a tensor of shape:  
`(batch_size, sequence_len, num_points, num_dims)`.  
Unbatched tensors are also allowed!  
  
The dataset (TeleopIcub) can be downloaded automatically if the files are not found in the given config path.  
For Human3.6M (H36M) please prepare your data following:  

TODO: HUMAN36M  

## Predictors
The trajectory prediction methods are organized as Predictor classes  
For example, the LinearPredictor class is the implementation of a simple Linear layer as a baseline for the task of Trajectory prediction  
Relying on the PytorhLightning Framework, it instantiates or load an existing torch Module, with a generic predictor wrapper for saving, loading, iterations over a sample and logging.  
  
Feel free to add some new predictors implementation following the example of this simple class.  

## Evaluator
We also provide a set of functions to run evalations and  plot some trajectories.  
Runners take a list of predictors, with a list of trajectories and provide an evaluation summary on the following metrics:
- Average Displacement Error(ADE)
- Final Displacement Error(FDE)
- Inference Time
TODO: ADD MORE


# Usage

Please look into the `example/` directory to find common usages of the API  
We use tensorboard for training logging, use `tensorboard --logdir {your_log_path}` to view the training infos (default log_path is `data/models/`)  


# Run tests

After installing, run this to make sure the installation is ok  

```bash
python -m unittest
```
