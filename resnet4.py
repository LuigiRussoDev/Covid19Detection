import keras
from math import ceil
import os
import cv2
import itertools
from sklearn.metrics import confusion_matrix
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from keras import optimizers
import pickle
from sklearn.metrics import classification_report
import skimage
from skimage.transform import resize

from tensorflow import keras
from tensorflow.keras import layers



from tqdm import tqdm

def lr_schedule(epochs):

    lr = 1e-3
    if epochs > 180:
        lr *= 0.5e-3
    elif epochs > 160:
        lr *= 1e-3
    elif epochs > 120:
        lr *= 1e-2

    elif epochs > 80:
        lr *= 1e-1

    print('Learning rate: ', lr)
    return lr


def Unit(x,filters,pool=False):
    res = x
    if pool:

        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        res = layers.Conv2D(filters=filters,kernel_size=[1,1],strides=(2,2),padding="same",

                     )(res)
    out = layers.BatchNormalization()(x)
    out = layers.Activation("relu")(out)
    out = layers.Conv2D(filters=filters, kernel_size=[3, 3], strides=[1, 1], padding="same",

                 )(out)

    out = layers.BatchNormalization()(out)
    out = layers.Activation("relu")(out)
    out = layers.Conv2D(filters=filters, kernel_size=[3, 3], strides=[1, 1], padding="same",
                 )(out)


    out = keras.layers.add([res,out])

    return out


#Define the model
def MiniModel(input_shape):


    images = keras.Input(input_shape)
    net = layers.Conv2D(filters=16, kernel_size=[7, 7],
                 strides=[2, 2], padding="same")(images)

    net = Unit(net, 16)
    net = Unit(net, 16)
    net = Unit(net, 16)
    net = Unit(net, 16)
    net = Unit(net, 16)
    net = Unit(net, 16)
    net = Unit(net, 16)
    net = Unit(net, 16)

    # [16]

    net = Unit(net, 32, pool=True)
    net = Unit(net, 32)
    net = Unit(net, 32)
    net = Unit(net, 32)
    net = Unit(net, 32)
    net = Unit(net, 32)
    net = Unit(net, 32)

    # 15

    net = Unit(net, 64, pool=True)
    net = Unit(net, 64)
    net = Unit(net, 64)
    net = Unit(net, 64)
    net = Unit(net, 64)
    net = Unit(net, 64)
    net = Unit(net, 64)
    net = Unit(net, 64)

    net = layers.BatchNormalization()(net)
    net = layers.Activation("relu")(net)

    #net = layers.Dropout(0.6)(net)

    net = keras.layers.AveragePooling2D(pool_size=(2,2))(net)
    net = layers.Flatten()(net)

    net = layers.Dense(units=3,activation="softmax")(net)

    model = keras.Model(inputs=images,outputs=net)

    return model



train_dir = "../data_3/train/"
test_dir =  "../data_3/test/"



def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure(10)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    y = np.repeat(np.arange(0, 3), 75)
    plt.xlim(-0.5, len(np.unique(y)) - 0.5)  # ADD THIS LINE
    plt.ylim(len(np.unique(y)) - 0.5, -0.5)  # ADD THIS LINE
    plt.savefig("confusion_matrix.png")

def get_data(folder):
    X = []
    y = []


    for folderName in os.listdir(folder):
        if not folderName.startswith('.'):
            if folderName in ['normal']:
                label = 0
            elif folderName in ['pneumonia']:
                label = 1
            elif folderName in ['covid']:
                label = 2
            else:
                label = 3
            for image_filename in tqdm(os.listdir(folder + folderName)):
                img_file = cv2.imread(folder + folderName + '/' + image_filename)
                if img_file is not None:
                    img_file = skimage.transform.resize(img_file, (32, 32, 3))
                    #img_file = scipy.misc.imresize(arr=img_file, size=(150, 150, 3))
                    img_arr = np.asarray(img_file)
                    X.append(img_arr)
                    y.append(label)

    X = np.asarray(X)
    y = np.asarray(y)
    return X,y



'''
train_x, train_y = get_data(train_dir)
test_x, test_y= get_data(test_dir)
np.save('train_x.npy', train_x)
np.save('train_y.npy', train_y)

np.save('test_x.npy', test_x)
np.save('test_y.npy', test_y)

'''

train_x = np.load("train_x.npy");
train_y = np.load("train_y.npy");
test_x = np.load("test_x.npy");
test_y = np.load("test_y.npy");





#normalize the data
train_x = train_x.astype('float32') / 255
test_x = test_x.astype('float32') / 255

#Subtract the mean image from both train and test set
train_x = train_x - train_x.mean()
test_x = test_x - test_x.mean()

#Divide by the standard deviation
train_x = train_x / train_x.std(axis=0)
test_x = test_x / test_x.std(axis=0)


datagen = ImageDataGenerator(rotation_range=10,
                             width_shift_range=5. / 32,
                             height_shift_range=5. / 32,
                             horizontal_flip=True)

# Compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied).
datagen.fit(train_x)


#Encode the labels to vectors
train_y = keras.utils.to_categorical(train_y, 3)
test_y = keras.utils.to_categorical(test_y,3)

#define a common unit
input_shape = (32,32,3)
model = MiniModel(input_shape)



model.summary()
#Specify the training components
sgd = optimizers.SGD(lr=0.0001, momentum=0.0, decay=0.0, nesterov=False)

epochs = 100
bs = 32
steps_per_epoch = ceil(460/bs)


model.compile(optimizer=keras.optimizers.Adam(),
              loss="categorical_crossentropy",metrics=['accuracy'])



# Fit the model on the batches generated by datagen.flow().
history = model.fit(train_x, train_y,
                    batch_size=bs,
                    epochs=epochs,
                    # We pass some validation for
                    # monitoring validation loss and metrics
                    # at the end of each epoch
                    validation_data=(train_x, train_y))



with open('from_net50.pickle', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
    #plt.show()
plt.savefig('grafico_accuracy_val_accuracy_resnet50.png')

plt.figure(2)
    # Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('grafico_loss_e_val_loss_resnet50.png')




#Evaluate the accuracy of the test dataset
accuracy  = model.evaluate(x=test_x,y=test_y,verbose=0)

target_names = ['Normal', 'Pneumonia','Covid']

y_pred = model.predict(test_x)

Y_true = np.argmax(test_y,axis = 1)

Y_pred_classes = np.argmax(y_pred, axis=1)


'''
print("y_pred ",y_pred)
print("y_True ",Y_true)
print("y_pred_classes ",Y_pred_classes)
'''


print('\n', classification_report(np.where(test_y > 0)[1], np.argmax(y_pred, axis=1),
                                  target_names=target_names))

confusion_mtx = confusion_matrix(Y_true, Y_pred_classes)

plot_confusion_matrix(confusion_mtx, classes = target_names)


print('Test loss:', accuracy[0])
print('Test accuracy:', accuracy[1])

model.save("covid19.h5")