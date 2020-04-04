# Covid19Detection
In this project I want to show the results about detections COVID19 through X-Ray Images using a DeepLearning approach. 
Although the data available are still scarce, I have tried to create a machine learning model through a neural network ResNet50. This model was made by myself and I did not make use of pretrained models available.
I have tried to make the project as personal as possible.

 ### Model Description
The model is a Convolutional Neural Network with shortcuts as showed in this paper: https://arxiv.org/pdf/1512.03385.pdf

### Libraries
For develop ResNet I have used Python as language programmin and libraries for machine learning as follows:
 - Keras 2.3
 - TensorFlow 2.0
 - Python 3.6
 - Matplotlib
 
### Dataset

As for the dataset I used the following: https://github.com/ieee8023/covid-chestxray-dataset
This dataset is the best known at the moment as regards the chest X-ray or CT images from Covid19. In particular, attention is paid to the repository is currently still "under construction" that is, it is still under construction. For more information, consult the dataset.

The dataset made available by the author pays attention to pneumonia diseases since COVID19 is a virus that attacks the lungs causing Pneumonia.
So the dataset mainly provides a subdivision between: Pneumonia and Covid.

Personally I opted to add more data to my dataset both in training and in testing in order to achieve greater reliability and to reach 70% for training and 30% for testing.
In summary I added more pneumonia data from the train and test dataset. Pneumonia data was delivered from the following dataset: https://www.kaggle.com/paultimothymooney/detecting-pneumonia-in-x-ray-images

Dataset: 
1. Train
    2. Pneumonia: 80
    3. COVID19: 80
    4. Normal: 80
2. Testing
    1. Pneumonia: 25
    2. COVID19: 25 
    3. Normal: 25

### Goal:
The goal proposed in this paper is to detect Covid and Non-Covid diseases.
Therefore I will try to make the model in question as perfect as possible by updating the dataset.

### Evaluations:
Accuracy: 0.94
Val Accuracy: 0.86
Val Loss: 0.34

![](img/confusion_matrix.png)







