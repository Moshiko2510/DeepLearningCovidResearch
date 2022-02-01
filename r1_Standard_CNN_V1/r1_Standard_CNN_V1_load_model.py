
from numpy.random import seed
seed(8)  # 1

import tensorflow
tensorflow.random.set_seed(7)

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

from tensorflow import keras
from tensorflow.keras.models import Model ,load_model

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np
import tensorflow as tf
from os import listdir

from keras.callbacks import ReduceLROnPlateau

from tensorflow.python.keras import models


def plot_confusion_matrix(cm,
                          target_names,
                          index,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()


data_list = listdir(r'C:\Users\moshi\OneDrive - ort braude college of engineering\Electrical Engineering\Research Project\folds2\fold1\train')

print(len(data_list))


IMAGE_SIZE = (150, 150)
NUM_CLASSES = len(data_list)
BATCH_SIZE = 8


train_list=[]
test_list=[]
acc_list=[]
loss_list=[]

for i in range(1):
    i=0
    if (i==0):
        DATASET_PATH = r'C:\Users\moshi\OneDrive - ort braude college of engineering\Electrical Engineering\Research Project\folds2\fold1\train'
        test_dir = r'C:\Users\moshi\OneDrive - ort braude college of engineering\Electrical Engineering\Research Project\folds2\fold1\test'
    elif (i==1):
            DATASET_PATH = r'C:\Users\moshi\OneDrive - ort braude college of engineering\Electrical Engineering\Research Project\folds2\fold2\train'
            test_dir = r'C:\Users\moshi\OneDrive - ort braude college of engineering\Electrical Engineering\Research Project\folds2\fold2\test'
    elif (i==2):
            DATASET_PATH = r'C:\Users\Daniel-PC\Desktop\folds2\Fold3\train'
            test_dir = r'C:\Users\Daniel-PC\Desktop\folds2\Fold3\test'
    elif (i==3):
            DATASET_PATH = r'C:\Users\Daniel-PC\Desktop\folds2\Fold4\train'
            test_dir = r'C:\Users\Daniel-PC\Desktop\folds2\Fold4\test'
    elif (i==4):
            DATASET_PATH = r'C:\Users\Daniel-PC\Desktop\folds2\Fold5\train'
            test_dir = r'C:\Users\Daniel-PC\Desktop\folds2\Fold5\test'

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    test_batches = test_datagen.flow_from_directory(test_dir,
                                                      target_size=IMAGE_SIZE,
                                                      shuffle=True,
                                                      batch_size=BATCH_SIZE,
                                                      #subset="validation",
                                                      seed=42,
                                                      class_mode="categorical"
                                                      )

    model=keras.models.load_model(r'C:\Users\moshi\OneDrive - ort braude college of engineering\Electrical Engineering\Research Project\results\r1_Standard_CNN_V1\model')

    import matplotlib.pyplot as plt


    #test_datagen = ImageDataGenerator(rescale=1. / 255)
    eval_generator = test_datagen.flow_from_directory(test_dir, target_size=IMAGE_SIZE, batch_size=1,
                                                      shuffle=False, seed=42, class_mode="categorical")
    pred_generator=eval_generator

    eval_generator.reset()

    x = model.evaluate(eval_generator,
                       steps=np.ceil(len(eval_generator)),
                       use_multiprocessing=False,
                       verbose=1,
                       workers=1,
                       )
    acc_list.append(x[1])
    loss_list.append(x[0])
    print('Test loss:', x[0])
    print('Test accuracy:', x[1])

    IMAGE_SIZE = (150, 150)



    from sklearn.metrics import confusion_matrix
    #from sklearn.metrics import plot_confusion_matrix
    from sklearn.metrics import classification_report

    filenames = eval_generator.filenames
    nb_samples = len(filenames)
    eval_generator.reset()
    predict = model.predict(eval_generator, steps=np.ceil(len(eval_generator)))

    predict = np.argmax(predict, axis=-1)
    classes = eval_generator.classes[eval_generator.index_array]
    acc = sum(predict == classes) / len(predict)

    names = ["covid", "normal", "pneumonia"]
    #print(confusion_matrix(classes, predict))
    cm = confusion_matrix(classes, predict)
    #print(cm)

    plot_confusion_matrix(cm=cm,
                          normalize=False,
                          target_names=names,
                          index=i+1,
                          title="Confusion Matrix")

    print(classification_report(classes, predict))
