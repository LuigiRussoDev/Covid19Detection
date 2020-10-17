from datetime import time

import keras
from math import ceil
import os
import cv2
import itertools
import sklearn
from sklearn.metrics import confusion_matrix
from scipy import interp
from keras import backend as K
import tensorflow as tf
from itertools import cycle
import numpy as np
from keras.layers import Layer
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from keras import optimizers
import pickle
from sklearn.metrics import classification_report
import skimage
from skimage.transform import resize
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from tensorflow import keras
from keras.layers import Input, MaxPooling2D, BatchNormalization, Activation, Conv2D, Flatten, Dense, add, Dropout, \
    AveragePooling2D
from keras.models import Model
from tqdm import tqdm


train_dir = "../data/data_big/train/"
test_dir = "../data/data_big/test/"

img_size = (32,32,3)

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


def Unit(x, filters, pool=False):
    res = x
    if pool:
        x = MaxPooling2D(pool_size=(2, 2))(x)
        res = Conv2D(filters=filters, kernel_size=[1, 1], strides=(2, 2), padding="same",

                     )(res)
    out = BatchNormalization()(x)
    out = Activation("relu")(out)
    out = Conv2D(filters=filters, kernel_size=[3, 3], strides=[1, 1], padding="same",

                 )(out)

    out = BatchNormalization()(out)
    out = Activation("relu")(out)
    out = Conv2D(filters=filters, kernel_size=[1, 1], strides=[1, 1], padding="same",
                 )(out)

    out = add([res, out])

    return out

class ODEBlock(Layer):

    def __init__(self, filters, kernel_size, **kwargs):
        self.filters = filters
        self.kernel_size = kernel_size
        super(ODEBlock, self).__init__(**kwargs)

    def build(self, input_shape):
        self.conv2d_w1 = self.add_weight("conv2d_w1", self.kernel_size + (self.filters + 1, self.filters), initializer='glorot_uniform')
        self.conv2d_w2 = self.add_weight("conv2d_w2", self.kernel_size + (self.filters + 1, self.filters), initializer='glorot_uniform')
        self.conv2d_b1 = self.add_weight("conv2d_b1", (self.filters,), initializer='zero')
        self.conv2d_b2 = self.add_weight("conv2d_b2", (self.filters,), initializer='zero')
        self.built = True
        super(ODEBlock, self).build(input_shape)

    def call(self, x):
        t = K.constant([0, 1], dtype="float32")
        return tf.contrib.integrate.odeint(self.ode_func, x, t, rtol=1e-3, atol=1e-3)[1]

    def compute_output_shape(self, input_shape):
        return input_shape

    def ode_func(self, x, t):
        y = self.concat_t(x, t)
        y = K.conv2d(y, self.conv2d_w1, padding="same")
        y = K.bias_add(y, self.conv2d_b1)
        y = K.relu(y)

        y = self.concat_t(y, t)
        y = K.conv2d(y, self.conv2d_w2, padding="same")
        y = K.bias_add(y, self.conv2d_b2)
        y = K.relu(y)

        return y

    def concat_t(self, x, t):
        new_shape = tf.concat(
            [
                tf.shape(x)[:-1],
                tf.constant([1],dtype="int32",shape=(1,))
            ], axis=0)

        t = tf.ones(shape=new_shape) * tf.reshape(t, (1, 1, 1, 1))
        return tf.concat([x, t], axis=-1)


# Define the model
def MiniModel(input_shape):
    images = Input(input_shape)

    net = Conv2D(16, kernel_size=[7, 7], strides=(1, 1), padding="same", activation='relu')(images)

    net = Unit(net, 8,pool=True)
    net = Unit(net, 8)
    net = Unit(net, 8) #Added more

    net = ODEBlock(8, (3, 3))(net)
    #net = Dropout(0.1)(net)
    net = AveragePooling2D(pool_size=(4, 4))(net)
    net = Flatten()(net)

    net = Dense(units=3, activation="softmax")(net)

    model = Model(inputs=images, outputs=net)

    return model


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
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


def roc_each_classes(test_y,y_pred):
    n_classes = 3

    # Plot linewidth.
    lw = 2

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(test_y[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(test_y.ravel(), y_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure(7)
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes ), colors):
        if i == 0:
            cl = "Normal"
        if i==1:
            cl = "Pneumonia"
        if i == 2:
            cl = "Covid"

        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f}) %s'
                       ''.format(i,roc_auc[i]) %cl )

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.savefig('Roc_each_classes.jpg')


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
                    img_file = skimage.transform.resize(img_file, img_size)
                    # img_file = scipy.misc.imresize(arr=img_file, size=(150, 150, 3))
                    img_arr = np.asarray(img_file)
                    X.append(img_arr)
                    y.append(label)

    X = np.asarray(X)
    y = np.asarray(y)
    return X, y


'''train_x, train_y = get_data(train_dir)
test_x, test_y= get_data(test_dir)

np.save('train_x.npy', train_x)
np.save('train_y.npy', train_y)

np.save('test_x.npy', test_x)
np.save('test_y.npy', test_y)'''


train_x = np.load("npy_32/train_x.npy");
train_y = np.load("npy_32/train_y.npy");
test_x = np.load("npy_32/test_x.npy");
test_y = np.load("npy_32/test_y.npy");


# normalize the data
train_x = train_x.astype('float32') / 255
test_x = test_x.astype('float32') / 255

train_x = train_x - train_x.mean()
test_x = test_x - test_x.mean()

train_x = train_x / train_x.std(axis=0)
test_x = test_x / test_x.std(axis=0)

datagen = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest")

datagen.fit(train_x,augment=True)

train_y = keras.utils.to_categorical(train_y, 3)
test_y = keras.utils.to_categorical(test_y, 3)


input_shape = img_size
model = MiniModel(input_shape)
model.summary()


sgd = optimizers.SGD(lr=0.0001, momentum=0.0, decay=0.0, nesterov=False)

epochs = 160
bs = 16
steps_per_epoch = ceil(2831 / bs)


model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])



history = model.fit(train_x, train_y,
                    batch_size=bs,
                    epochs=epochs,
                    validation_data=(test_x, test_y))

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
# plt.show()
plt.savefig('plot_accuracy_odenet.png')

plt.figure(2)
# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('plot_loss_odenet.png')

# Evaluate the accuracy of the test dataset
accuracy = model.evaluate(x=test_x, y=test_y, verbose=0)

target_names = ['Normal', 'Pneumonia', 'Covid']

y_pred = model.predict(test_x)
Y_true = np.argmax(test_y, axis=1)
Y_pred_classes = np.argmax(y_pred, axis=1)
roc_each_classes(test_y,y_pred)


print('\n', classification_report(np.where(test_y > 0)[1], np.argmax(y_pred, axis=1),
                                  target_names=target_names))

confusion_mtx = confusion_matrix(Y_true, Y_pred_classes)

plot_confusion_matrix(confusion_mtx, classes=target_names)

print('Test loss:', accuracy[0])
print('Test accuracy:', accuracy[1])

model.save("covid19.h5")