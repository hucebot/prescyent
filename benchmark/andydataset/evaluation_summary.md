# Task description
We will perform short to long-term forecasting of human motion based solely on previous observations.  

## Dataset
We used the AndyDataset, with its 195 Trajectories split into train, test, and val sets with ratios of 0.8, 0.15, and 0.05 respectively.  
A trajectory is a sequence of 23 joints' positions and orientations of a human performing various tasks in a closed environment (please refer to the paper for more details).  
For each joint, we have its absolute position in meters on axes x, y, z, and absolute rotation in quaternions qx, qy, qz, qw.  

From these, we generate x, y pairs using the following dataset configuration:  
All joint positions are set relative to the pelvis position.  
The dataset is downsampled to 10Hz.  
The input sequence size is 25 frames (2.5s history).  
We will have a varying output sequence size between 1 and 10 (0.1s to 1s future).  
All 23 joints' features will be used for the input.  
Only the features of the RightHand are predicted.  
Inputs and output features are:  
    - x, y, z positions relative to the pelvis for the given joints (dim=3).  
    - Absolute rotation converted to continuous 6d representation for 3D rotation using a Gram-Schmidt-like process (see reference for more details).  

So our tensors have shape:  
 - Input tensor has shape (25, 23, 9)  
 - Output tensor has shape (f, 1, 9)  

For our task, we will vary f among [1, 3, 5, 8, 10], for a sequence prediction range between 100ms and 1000ms.

## Predictors
We trained and compared 3 predictors with various future size over the AndyDataset, using MSE loss.  
MLP predictors with 8 fully connected layers, hidden size of 128, and RELU activation function. Its inputs are preprocessed with 'batch_normalization' and 'norm_on_last_input' (all features of the input sequence are relative to the features of the last frame of the input).  
The best-trained versions of siMLPe (see paper in reference for more details) predictors all had 'spatial_normalization' and 'dct' activated. Other parameters' values like 'num_layer', 'hidden_size', and 'norm_on_last_input' vary depending on the future size we trained on. (I invite you to check directly the config file of trained models to see these parameters).  
The Constant Predictor is used as a baseline and requires no training. We simply output the last frame of input for the requested future.

# Evaluation results
Here we show the results of the best trained model of each predictor.  
Theses results where optained unsing version 0.3.0 of the library.  

## Coordinate
Average prediction error for Coordinate in mm at given frames. Lower is better.  
|Future_size|100ms|300ms|500ms|800ms|1000ms|
|:---|---:|---:|---:|---:|---:|
|ConstantPredictor|18.84|50.92|76.85|108.23|126.24|
|MlpPredictor|11.12|32.36|51.17|75.40|86.87|
|siMLPe|**11.02**|**30.48**|**47.60**|**69.78**|**80.52**|

## Rotation
Average prediction error for Rotation in degrees at given frames. Lower is better.  
|Future_size|100ms|300ms|500ms|800ms|1000ms|
|:---|---:|---:|---:|---:|---:|
|ConstantPredictor|5.87|14.06|20.38|27.71|31.83|
|MlpPredictor|4.24|10.37|14.84|19.61|22.23|
|siMLPe|**3.62**|**9.27**|**13.34**|**17.85**|**19.99**|


# Should we train for a specific future size?
We will compare the results of models trained for a specific future size with the performance of the model for the longest future size at each time step of its predicted sequence.  
Here we display the results at each time frame from the best model for each predictor trained on a 1000ms future.

## Coordinate
Average prediction error for Coordinate in mm at given frames. Lower is better.  
|Error at future frame|100ms|300ms|500ms|800ms|1000ms|
|:---|---:|---:|---:|---:|---:|
|ConstantPredictor|18.84|50.92|76.85|108.23|126.24|
|MlpPredictor|**15.24**|37.19|52.83|72.83|86.87|
|siMLPe|21.78|**36.88**|**51.79**|**68.98**|**80.52**|

## Rotation
Average prediction error for Rotation in degrees at given frames. Lower is better.
|Error at future frame|100ms|300ms|500ms|800ms|1000ms|
|:---|---:|---:|---:|---:|---:|
|ConstantPredictor|5.87|14.06|20.38|27.71|31.83|
|MlpPredictor|5.07|10.95|15.05|19.60|22.23|
|siMLPe|**4.85**|**10.27**|**14.09**|**17.77**|**19.99**|

## Conclusion
Comparing these results with the previous pair of tables shows us a clear gain in the short-term results (<=500ms) and would incline us to have a future_size in training that is the closest possible to the final use case. Better results on the last frame don't assure the best sequence is outputted.  
Also, our 1000ms models perform better on the 800ms prediction than the 800ms models. This should be investigated further, on how we should define the latest frame we want to predict; maybe a model trained on 1200ms would perform better on a 1000ms prediction than a model for which it is the last frame?  


# References

siMLPe  
Guo, W., Du, Y., Shen, X., Lepetit, V., Alameda-Pineda, X., & Moreno-Noguer, F. (2022, July 4). Back to MLP: a simple baseline for human motion prediction. arXiv.org. https://arxiv.org/abs/2207.01567  

AndyDataset  
Maurice P., Malaisé A., Amiot C., Paris N., Richard G.J., Rochel O., Ivaldi S. « Human Movement and Ergonomics: an Industry-Oriented Dataset for Collaborative Robotics ». The International Journal of Robotics Reserach, Volume 38, Issue 14, Pages 1529-1537.  

On the Continuity of Rotation Representations in Neural Networks  
Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H. (2018, December 17). On the Continuity of Rotation Representations in Neural Networks. arXiv.org. https://arxiv.org/abs/1812.07035