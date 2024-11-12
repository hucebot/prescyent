
# Get Started
PreScyent is a trajectory forecasting library, built upon pytorch_lightning  
It comes with datasets such as:
- [AndyData-lab-onePerson](https://zenodo.org/records/3254403#.Y_9fwBeZMVk)  
- [AndyData-lab-onePersonTeleoperatingICub](https://zenodo.org/record/5913573)  
- [Human3.6M](http://vision.imar.ro/human3.6m/description.php)  
- And synthetics dataset to test simple properties of our predictors.  

It come also with baselines to run over theses datasets, refered in the code as Predictors, such as:
- [SiMLPe](https://arxiv.org/abs/2207.01567) a MultiLayer Perceptron (MLP) with Discrete Cosine Transform (DCT), shown as a strong baseline acheiving SOTA results against bigger and more complicated models.  
- [Seq2Seq](https://arxiv.org/abs/1409.3215), an architecture mapping an input sequence to an output sequence, that originated from NLP and grew in popularity for time series predictions. Here we implemented an RNN Encoder and RNN Decoder.  
- Probabilistic Movement Primitives (ProMPs), an approach commonly used in robotics to model movements by learning from demonstrations and generating smooth, adaptable trajectories under uncertainty.  
- Some simple ML Baselines such as a Fully Connected MultiLayer Perceptron and an autoregressive predictor with LSTMs.  
- Non machine learning baselines, maintaining the velocity or positions of the inputs, or simply delaying it.  


## Task Definition

- With **H** our history size, the number of observed frames​  
- With **F** our future size, the number of frames we want to predict​  
- With **X** the points and features used as input​  
- With **Y** the points and features produced as output​  
We define **P** our predictor such as, at a given timestep **T** we have:​  


/TODO
$$
P(X_{T-H}, \dots, X_T) = Y_{T+1}, \dots, Y_{T+F}
$$​

## Installation

### From pypi
You can install released versions of the project from [PyPi](https://pypi.org/project/prescyent/). install using pip from source (you may want to be in a [virtualenv](https://python-guide-pt-br.readthedocs.io/fr/latest/dev/virtualenvs.html) beforehand):  
```bash
pip install prescyent
```
### From Docker
You can build an image docker from the Dockerfile at the source of the repository.  
Please refer to [docker documentation](https://docs.docker.com) for build command and options.  
The Dockerfile is designed to be run interactively.  

### From source
If you want to setup a dev environment, you should clone the repository:  

```bash
git clone git@github.com:hucebot/prescyent.git
cd prescyent
```
Then install using pip from source (you may want to be in a [virtualenv](https://python-guide-pt-br.readthedocs.io/fr/latest/dev/virtualenvs.html) beforehand):  
For dev install (recommended if you intent to add new classes to the lib) use:  
```bash
pip install -e .
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
Then when creating an instance of a Dataset, make sure you pass the path to the newly generated hdf5 file to the dataset's config attribute `hdf5_path`.  

#### TeleopIcubDataset
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

#### H36MDataset
For [Human3.6M](http://vision.imar.ro/human3.6m/description.php) you need to download the zip [here](http://www.cs.stanford.edu/people/ashesh/h3.6m.zip) and prepare your data following this directory structure:  
```bash
└── h36m
|   ├── S1
|   ├── S5
|   ├── S6
|   ├── ...
|   ├── S11
```

#### AndyDataset
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
- Real Time Factor (RTF)

## Usage

### Examples and tutorials
Please look into the `examples/` directory to find common usages of the library  
We use tensorboard for training logging, use `tensorboard --logdir {log_path}` to view the training and testing infos (default log_path is `data/models/`)  

For example to run the script for mlp training on teleopIcub dataset use this in the enviroment where prescyent is installed:
```bash
python examples/mlp_icub_train.py
```  
If you want to start a training from a config file (as examples/configs/mlp_teleopicub_with_center_of_mass_context.json), use the following:  
```bash
python examples/train_from_config.py examples/configs/mlp_teleopicub_with_center_of_mass_context.json
```  

The script `start_multiple_trainings.py` is an example of how to generate variations of configuration files and running them using methods from `train_from_config.py`  

Also for evaluation purposes, you can see an example running tests and plots using AutoPredictor and AutoDataset in `load_and_plot_predictors.py`  


### Extend the lib with a custom dataset or predictor
Predictors inherit from the BasePredictor class, which define interfaces and core methods to keep consistency between each new implementation.  
Each Predictor defines its PredictorConfig with arguments that will be passed on to the core class, again with a BaseConfig with common attributes that needs to be defined.  

In the same way you can extend the dataset module with a new Dataset inheriting from TrajectoriesDataset with its own DatasetConfig.  
If you simply want to test a Predictor over some data of yours, you can create an instance of CustomDataset. As long as you turned your lists of episodes into Trajectories, the CustomDataset allows you to split them into training samples using a generic DatasetConfig and use all the functionnalties of the library as usual (except that a CustomDataset cannot be loaded using AutoDataset)...  
