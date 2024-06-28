# Getting started with PreScyent
PreScyent is a trajectory forecasting library, built upon pytorch_lightning  
It comes with datasets such as:
- [AndyData-lab-onePersonTeleoperatingICub](https://zenodo.org/record/5913573)  
- [AndyData-lab-onePerson](https://andydataset.loria.fr/)  
- [Human3.6M](http://vision.imar.ro/human3.6m/description.php)  

And baselines to run over theses datasets, refered in the code as Predictors, such as:
- [SiMLPe](https://arxiv.org/abs/2207.01567)


## Installation

### From Pypi
Coming soon.  

### From Docker
You can build an image docker from the Dockerfile at the source of the repository.  
Please refer to [docker documentation](https://docs.docker.com) for build command and options.  
The Dockerfile is designed to be run interactively.  

### From source
Clone the repository and cd  

```bash
git clone git@github.com:hucebot/prescyent.git
cd prescyent
```
Then install using pip from source (you may want to be in a [virtualenv](https://python-guide-pt-br.readthedocs.io/fr/latest/dev/virtualenvs.html) beforehand):  
For dev install (recommended if you intent to add new classes into the lib and contribute) use:  
```bash
pip install -e .
```
Otherwise simply use:  
```bash
pip install .
```

## Usages

You want to train a new model over a 

## Tutorials and examples
Find tutorials and examples in the github repo [here](https://github.com/hucebot/prescyent/tree/main/examples)
