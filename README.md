# PreScyent
Data-driven trajectory prediction library (python)  

# Installation
Clone repository then run this from folder  

```bash
pip install .
```

# Get Started

PreScyent is a forecasting library, built uppon pytorch_lightning  
It comes with datasets such as AndyData-lab-onePersonTeleoperatingICub  
And methods to perform trajectory prediction on this kind of datasets  
  
Theses methods are organised as predictors classes  
For example, the LSTMPredictor class is the implementation of a LSTM for the task of Trajectory prediction  
It overrides the following methods:  

```python

```

Feel free to add some new implementation following the example of this class for now  

# Usage

Please look into the `sample/` directory to find common usages of the API  

# Run tests

After installing, run this to make sure the installation is ok  
```bash
python -m unittest
```
