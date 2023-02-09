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
And methods to perform trajectory prediction on this kind of datasets  
  
Theses methods are organised as predictors classes  
For example, the LinearPredictor class is the implementation of a simple Linear layer as a baseline for the task of Trajectory prediction  
It instanciates both a custom torch Module and Ligtning Module with a specific configuration:  
Feel free to add some new predictors implementation, with its modules following the example of this class for now  

# Usage

Please look into the `sample/` directory to find common usages of the API  
We use tensorboard for training logging, use `tensorboard --logdir {your_log_path}` to view the training infos (default log_path is `data/models/`)  

# Run tests

After installing, run this to make sure the installation is ok  

```bash
python -m unittest
```
