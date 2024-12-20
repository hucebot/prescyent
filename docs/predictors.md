# Predictors
Our architecture used to perform predictions over trajectories.  
They inherits from BasePredictor, implements its own .predict() method, along other methods to log, save, load, train and test !  

Import all of theses predictors and their configs from `prescyent.predictor`.  


## Algorithmic Baselines
Theses Predictors have very basic behavior and config, they are used to compare results with some simple intuitions while they share the same Predictor API as any more complex model.  

### ConstantPredictor
The ConstantPredictor returns a sequence of size F = future_size, with the last observed frame as the whole sequence of output  
It behaves like this:  
$P(X_{T-H}, \dots, X_T) = X_T, \dots, X_T$  

[Config](configuration_files.rst#predictorconfig)

### ConstantDerivativePredictor
The ConstantDerivativePredictor returns a sequence of size F = future_size, with the last observed frame returned for each predicted frame
It behaves like this:  
$P(X_{T-H}, \dots, X_T) = X_T + (X_T - X_{T-1}), \dots, X_{T+F-1} + (X_T - X_{T-1})$  

[Config](configuration_files.rst#predictorconfig)

### DelayedPredictor
The DelayedPredictor returns a sequence of size F = future_size, with the input as the output  
(If future_size >= history_size, it behaves like the constant predictor, else it behaves like this):  
$P(X_{T-H}, \dots, X_T) = X_{T-H}, \dots, X_T$  

[Config](configuration_files.rst#predictorconfig)

## ProMPs
Probabilistic Movement Primitives (ProMPs), an approach commonly used in robotics to model movements by learning from demonstrations and generating smooth, adaptable trajectories under uncertainty.  

[Config](configuration_files.rst#prompconfig)

## ML Models

### SiMLPe
A MultiLayer Perceptron (MLP) with Discrete Cosine Transform (DCT), shown as a strong baseline achieving SOTA results against bigger and more complicated models.  

[Paper](https://arxiv.org/abs/2207.01567)  
[Config](configuration_files.rst#simlpeconfig)  

### Seq2Seq
An architecture mapping an input sequence to an output sequence, that originated from NLP and grew in popularity for time series predictions. Here we implemented an RNN Encoder and RNN Decoder.  

[Paper](https://arxiv.org/abs/1409.3215)  
[Config](configuration_files.rst#seq2seqconfiguration_files.rst#sarlstmconfig)  

### MLP
Simple ML Baselines consisting of a configurable Fully Connected MultiLayer Perceptron.  
It's a simple architecture you can use as an example for sequence to sequence training and quick tests.  

[Config](configuration_files.rst#mlpconfig)

### SARLSTM
Simple ML Baselines consisting of an autoregressive architecture with LSTMs.  
It's an architecture you can use as an example for an auto regressive training.  
This model requires x, y pairs that differ from classical sequence to sequence training, build your dataset using an auto regressive [LearningType](enums.rst#learningtypes).  

[Config](configuration_files.rst#sarlstmconfig)  
