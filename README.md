# Covid19Detection
In this project I want to show the results about detections COVID19 through X-Ray Images using a DeepLearning approach. 
Although the data available are still scarce, I have tried to create a machine learning model through a neural network ResNet50. This model was made by myself and I did not make use of pretrained models available.
I have tried to make the project as personal as possible.

 ### Model Description
The model is a Convolutional Neural Network with shortcuts as showed in this paper: https://arxiv.org/pdf/1512.03385.pdf

### Libraries
My ResNet50 model was developed with the Python programming language and the following libraries:
 - Keras 2.3
 - TensorFlow 2.0
 - Python 3.6
 - Matplotlib
 
### Dataset

About dataset I used the following: https://github.com/ieee8023/covid-chestxray-dataset
This dataset is the best known at the moment as regards the chest X-ray images from Covid19. In particular, attention is paid to the repository is currently still "working in progress".

The dataset made available by the author pays attention to pneumonia diseases since COVID19 is a virus that attacks the lungs causing Pneumonia.
So the dataset mainly provides a subdivision between: Pneumonia cases (MERS, SARS, and ARDS) and COVID19.
The data are few in previous repository. 

I opted to add more data to my dataset both in train and in test in order to achieve much performance and to reach 70% for training and 30% for testing in case of Pneumonia cases. 
Another class was used. In this case I have used class "normal" to indentify a good status of healthy.
The added class "Normal" and the additional data for penumonia cases are available at the follow link:
https://www.kaggle.com/paultimothymooney/detecting-pneumonia-in-x-ray-images
The images between the two datasets are both X-RAY.


Dataset: 
1. Train
    1. Pneumonia: 70
    2. COVID19: 70
    3. Normal: 70
2. Testing
    1. Pneumonia: 30
    2. COVID19: 30
    3. Normal: 30

### Goal:
The goal proposed in this paper is to detect Covid and Non-Covid diseases.
Therefore I will try to make the model in question as perfect as possible by updating the dataset.

### How to use:
In order to avoid always uploading the images before training I uploaded the data located in the "data" as numpy array. 
If you want download dataset, first you need to download repository: https://github.com/ieee8023/covid-chestxray-dataset and then run:

```sh
$ mkdir -p train/covid19
$ mkdir -p train/pneumonia
$ mkdir -p test/covid19
$ mkdir -p test/pneumonia
$ python create_dataset.py
```

The "Normal" class is avaiable at the follow dataset: https://www.kaggle.com/paultimothymooney/detecting-pneumonia-in-x-ray-images

Don't forget what has been said previously. That is, if you want more data you must also use the previous dataset and add more data for pneumonia in order to have a 70% - 30% dataset.



To train model you should run:
```sh
$ python resnet2.py
```
### Hyperparameter
For the training the following hyperparameters are used: 
- AdamOptimization in schedule
- 100 Epochs
- BatchSize 16
- Last layer a softmax is used
- Loss function Cross Entropy Categorical is used 
- Learning rate in schedule


### Results:
- Accuracy: 0.94
- Val Accuracy: 0.86
- Val Loss: 0.34

Results | #Precision | #Recall | #f1-score | #support 
--- | --- | --- | --- |--- 
Normal | 0.97 | 0.97 | 0.97 | 31 
Pneumonia | 0.33 | 0.60 | 0.43 | 5 
COVID19 | 0.93 | 0.81 | 0.86 | 31 

![](img/confusion_matrix.png)

As we can see there is a lot of "miss classification" when we compare an image XRay Pneumonia is also classified as COVID. This occurs COVID looks like Pneumonia just like the Pneumonia dataset. 

### Conclusion
My goal is always to increase the dataset of images from COVID and to take care of the accuracy of my model.
Through the confusion matrix it has been possible to observe that COVID images are often confused by the Pneumonia class which includes various pneumonia diseases. This "Miss classification" phenomenon is caused by the fact that COVID manifests itself as a form of Pneumonia just like the data in "Pneumonia".

### Contact
My contact is: luigi.russo@studenti.uniparthenope.it







