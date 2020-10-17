# Covid19Detection
In this project I want to show the results about detections COVID19 through X-Ray Images using a DeepLearning approach. 

 ### Model Description
The Neural Network used for this work is a Odenet neural network: https://arxiv.org/abs/1806.07366
In particular I used a Res-Ode that's mean a miniblock of resnet before the ODE block. 

### Libraries
The main machine learning libraries used for this work are:
 - Keras 2.3
 - TensorFlow 2.0
 - Python 3.6
 - Matplotlib
 
### Dataset
For this work I've used the following dataset for image covid by x-ray: https://data.mendeley.com/datasets/8h65ywd2jr/3 
The other two classes were generated by following dataset: https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia
This case I got two more classes (Normal and Pneumonia images by x-ray). 
The dataset has been splitted as follows:
1. Train (2831 images for each)
   1. Normal: 
   1. Covid
   1. Pneumonia

1. Test (1213 images for each)
   1. Normal
   1. Covid
   1. Pneumonia
 
### Goal:
The goal proposed in this paper is to detect Covid and Non-Covid diseases.
Therefore I will try to make the model as accurate as possible. 

### How to use:
As you can see the dataset is composed by several data taken from web. (This practice has also been widely used by various authors in the realization of the same papers about Covid with deep NN). 
Before to train your model you have to set path for dataset. It is advisable to upload images only once and store them in an array (in our case you can find array in "data" folder).
In my case I've stored data in npy_32 (where 32 means size of images uploaded). 

To train run: 
```sh
$ python odenet_big.py
```

### Hyperparameter
For the training the following hyperparameters are used: 
- AdamOptimization in schedule
- 160 Epochs
- BatchSize 16
- Last layer a softmax is used
- Loss function Cross Entropy Categorical is used 
- Learning rate in schedule from 0.001
- BatchNorm 


### Results:
- Accuracy: 0.99
- Val Accuracy: 0.95
- Val Loss: 0.94

Results | #Precision | #Recall | #f1-score | #support 
--- | --- | --- | --- |--- 
Normal | 0.97 | 0.95 | 0.96 | 213 
Pneumonia | 0.96 | 0.93 | 0.94 | 218 
COVID19 | 0.96 | 1.0 | 0.98 | 219

![](img/confusion_matrix.png)


![](img/Roc_each_classes.jpg)


### Conclusion
My goal is always to increase the dataset of images from COVID and to take care of the accuracy of my model.
Through the confusion matrix it has been possible to observe that COVID images are often confused by the Pneumonia class which includes various pneumonia diseases. This "Miss classification" phenomenon is caused by the fact that COVID manifests itself as a form of Pneumonia just like the data in "Pneumonia".

### Contact
My contact is: luigi.russo@studenti.uniparthenope.it







