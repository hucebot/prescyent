<p align="center">
    <img alt="PreScyent" src="https://raw.githubusercontent.com/hucebot/prescyent/main/assets/logo.png">
</p>

<h2 style="text-align: center;">
Data-driven trajectory forecasting library built in python
</h2>

<p align="center" width="100%">
    <img alt="Trajectory plot" src="https://raw.githubusercontent.com/hucebot/prescyent/main/assets/mlp_icub_test_plot.png" width="48%" >
    <img alt="Trajectory visualization" src="https://raw.githubusercontent.com/hucebot/prescyent/main/assets/S5_greeting_1_animation.gif" width="40%">
</p>

# Get Started
PreScyent is a trajectory forecasting library, built upon pytorch_lightning  
It comes with datasets such as:
- [AndyData-lab-onePerson](https://zenodo.org/records/3254403#.Y_9fwBeZMVk)  
- [AndyData-lab-onePersonTeleoperatingICub](https://zenodo.org/record/5913573)  
- [Human3.6M](http://vision.imar.ro/human3.6m/description.php)  

And methods to perform trajectory prediction on this kind of datasets  

## Installation

### From pipy
You can install released versions of the project from [PyPi](https://pypi.org/project/prescyent/). install using pip from source (you may want to be in a [virtualenv](https://python-guide-pt-br.readthedocs.io/fr/latest/dev/virtualenvs.html) beforehand):  
```bash
pip install prescyent
```
### From Docker
You can build an image docker from the Dockerfile at the source of the repository.  
Please refer to [docker documentation](https://docs.docker.com) for build command and options.  
The Dockerfile is designed to be run interactively.  

### From source
Clone the repository:  

```bash
git clone git@github.com:hucebot/prescyent.git
cd prescyent
```
Then install using pip from source (you may want to be in a [virtualenv](https://python-guide-pt-br.readthedocs.io/fr/latest/dev/virtualenvs.html) beforehand):  
For dev install (recommended if you intent to add new classes to the lib) use:  
```bash
pip install -e .
```
Otherwise simply use:  
```bash
pip install .
```

## Datasets
Each dataset is composed a list Trajectories, splitted as train, test and val.  
In this lib, we call "Trajectory" a sequence in time splitted in F frames, tracking P points above D dimensions.  
It is represented with a batched tensor of shape:  
`(B, F, P, D)`.  
Note that unbatched tensors are also allowed for inference.  
For each Trajectory we describe its tensor with Features, a list of Feature describing what are the dimensions trasked for each point at each frame. Theses allows conversions to occur in background or as preprocessing, as well as using some distance specific losses and metrics (e.g. euclidian distance for Coordinates and geodesic distance for Rotations)  
Alongside each trajectory tensor, some dataset provide some additional "context" (images, center of mass, velocities...), that is represented inside the library as a dictionary of tensors.  

### Downloads
We use HDF5 file format to load a dataset internally. Please get the original data and pre process them using the scripts in `/datapreprocessing` to match the library's format.  
Then when creating an instance of a Dataset, make sure you pass the path to the newly generated hdf5 file to teh dataset's config attribute `hdf5_path`.  

Download AndyData-lab-prescientTeleopICub's data [here](https://zenodo.org/record/5913573/)
and unzip it, it should be following this structure:  
```bash
├── AndyData-lab-prescientTeleopICub
│   └── datasetMultipleTasks
│       └── AndyData-lab-prescientTeleopICub
│           ├── datasetGoals
│           │   ├── 1.csv
│           │   ├── 2.csv
│           │   ├── 3.csv
│           │   ├── ...
│           ├── datasetObstacles
│           │   ├── 1.csv
│           │   ├── 2.csv
│           │   ├── 3.csv
│           │   ├── ...
│           ├── datasetMultipleTasks
│           │   ├── BottleBox
│           │   │   ├── 1.csv
│           │   │   ├── 2.csv
│           │   │   ├── 3.csv
│           │   │   ├── ...
│           │   ├── BottleTable
│           │   │   ├── 1.csv
│           │   │   ├── 2.csv
│           │   │   ├── 3.csv
│           │   │   ├── ...
│           │   ├── ...

```

For [Human3.6M](http://vision.imar.ro/human3.6m/description.php) you need to download the zip [here](http://www.cs.stanford.edu/people/ashesh/h3.6m.zip) and prepare your data following this directory structure:  
```bash
└── h36m
|   ├── S1
|   ├── S5
|   ├── S6
|   ├── ...
|   ├── S11
```

For [AndyDataset](https://andydataset.loria.fr/) you need to download the zip [here](https://zenodo.org/records/3254403/files/xens_mnvx.zip?download=1) and prepare your data following this directory structure:  
```bash
└── AndyData-lab-onePerson
|   └── xsens_mnvx
|       ├── Participant_541
|       ├── Participant_909
|       ├── Participant_2193
|       ├── ...
|       ├── Participant_9875
```

## Predictors
The trajectory prediction methods are organized as Predictor classes.  
For example, the MlpPredictor class is the implementation of a configurable MLP as a baseline for the task of Trajectory prediction.  
Relying on the PytorchLightning Framework, it instantiates or load an existing torch Module, with a generic predictor wrapper for saving, loading, iterations over a sample and logging.  
Feel free to add some new predictor implementations following the example of this simple class, inheriting at least from the BasePredictor class.  

## Evaluator
We also provide a set of functions to run evaluations and plot some trajectories.  
Runners take a list of predictors, with a list of trajectories and provide an evaluation summary on the following metrics:
- Average Displacement Error (ADE)
- Final Displacement Error (FDE)
- Mean Per Joint Position Error (MPJPE)
- Inference Time

# Usage
Please look into the `examples/` directory to find common usages of the library  
We use tensorboard for training logging, use `tensorboard --logdir {log_path}` to view the training and testing infos (default log_path is `data/models/`)  
For example to run the script for mlp training on teleopIcub dataset use:
```bash
python examples/mlp_icub_train.py
```
If you want to update start a training from a config file (as examples/configs/mlp_h36m.json), use the following:
```bash
python examples/train_from_config.py examples/configs/mlp_h36m.json
```
## Ros2
See [dedicated repository](https://github.com/hucebot/ros2_prescyent/tree/dev)

## Run tests
After installing, run this to make sure the installation is ok  

```bash
python -m unittest -v
```

# Extend the lib with a custom dataset or predictor
Predictors inherit from the BasePredictor class, which define interfaces and core methods to keep consistency between each new implementation.  
Each Predictor defines its PredictorConfig with arguments that will be passed on to the core class, again with a BaseConfig with common attributes that needs to be defined.  

# References
siMLPe  
Wen Guo, Yuming Du, Xi Shen, Vincent Lepetit, Xavier Alameda-Pineda, et al.. Back to MLP: A Simple Baseline for Human Motion Prediction. WACV 2023 - IEEE Winter Conference on Applications of Computer Vision, Jan 2023, Waikoloa, United States. pp.1-11. ⟨hal-03906936⟩​  

AndyDataset  
Maurice P., Malaisé A., Amiot C., Paris N., Richard G.J., Rochel O., Ivaldi S. « Human Movement and Ergonomics: an Industry-Oriented Dataset for Collaborative Robotics ». The International Journal of Robotics Reserach, Volume 38, Issue 14, Pages 1529-1537.  

TeleopIcub Dataset  
Penco, L., Mouret, J., & Ivaldi, S. (2021, July 2). Prescient teleoperation of humanoid robots. arXiv.org. https://arxiv.org/abs/2107.01281  

H36M Dataset  
Human3.6M: Large scale datasets and predictive methods for 3D human sensing in natural environments. (n.d.). IEEE Journals & Magazine | IEEE Xplore. https://ieeexplore.ieee.org/document/6682899  

On the Continuity of Rotation Representations in Neural Networks  
Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H. (2018, December 17). On the Continuity of Rotation Representations in Neural Networks. arXiv.org. https://arxiv.org/abs/1812.07035  
