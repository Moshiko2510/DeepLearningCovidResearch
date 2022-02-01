
from numpy.random import seed
seed(8)  # 1

import tensorflow
tensorflow.random.set_seed(7)

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

from tensorflow.keras.applications import Xception
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model ,load_model
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np
import tensorflow as tf
from os import listdir



from tensorflow.python.keras import models
from tensorflow.python.keras import layers

from tensorflow.keras import optimizers


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


data_list = listdir(r'C:\Users\Daniel-PC\Desktop\folds2\Fold1\train')

print(len(data_list))


IMAGE_SIZE = (150, 150)
NUM_CLASSES = len(data_list)
BATCH_SIZE = 8
NUM_EPOCHS = 12
LEARNING_RATE = 0.0001

train_list=[]
test_list=[]
acc_list=[]
loss_list=[]

for i in range(1):
    i=4
    if (i==0):
        DATASET_PATH = r'C:\Users\Daniel-PC\Desktop\folds2\Fold1\train'
        test_dir = r'C:\Users\Daniel-PC\Desktop\folds2\Fold1\test'
    elif (i==1):
            DATASET_PATH = r'C:\Users\Daniel-PC\Desktop\folds2\Fold2\train'
            test_dir = r'C:\Users\Daniel-PC\Desktop\folds2\Fold2\test'
    elif (i==2):
            DATASET_PATH = r'C:\Users\Daniel-PC\Desktop\folds2\Fold3\train'
            test_dir = r'C:\Users\Daniel-PC\Desktop\folds2\Fold3\test'
    elif (i==3):
            DATASET_PATH = r'C:\Users\Daniel-PC\Desktop\folds2\Fold4\train'
            test_dir = r'C:\Users\Daniel-PC\Desktop\folds2\Fold4\test'
    elif (i==4):
            DATASET_PATH = r'C:\Users\Daniel-PC\Desktop\folds2\Fold5\train'
            test_dir = r'C:\Users\Daniel-PC\Desktop\folds2\Fold5\test'

    # Train datagen here is a preprocessor
    train_datagen = ImageDataGenerator(rescale=1. / 255,
                                       rotation_range=50,
                                       featurewise_center=True,
                                       featurewise_std_normalization=True,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       shear_range=0.25,
                                       zoom_range=0.1,
                                       zca_whitening=True,
                                       channel_shift_range=20,
                                       horizontal_flip=True,
                                       vertical_flip=True,
                                       # validation_split = 0.2,
                                       fill_mode='constant')

    test_datagen = ImageDataGenerator(rescale=1. / 255)


    train_batches = train_datagen.flow_from_directory(DATASET_PATH,
                                                      target_size=IMAGE_SIZE,
                                                      shuffle=True,
                                                      batch_size=BATCH_SIZE,
                                                      #subset="training",
                                                      seed=42,
                                                      class_mode="categorical"
                                                      )

    test_batches = test_datagen.flow_from_directory(test_dir,
                                                      target_size=IMAGE_SIZE,
                                                      shuffle=True,
                                                      batch_size=BATCH_SIZE,
                                                      #subset="validation",
                                                      seed=42,
                                                      class_mode="categorical"
                                                      )

    train_list.append(train_batches)
    test_list.append(test_batches)

    # our model:
    conv_base = Xception(weights='imagenet',
                         include_top=False,
                         input_shape=(150, 150, 3))

    conv_base.trainable = True

    model = models.Sequential()
    model.add(conv_base)

    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(3, activation='softmax'))

    model.compile(loss='categorical_crossentropy',  # for multiclass use categorical_crossentropy
                  optimizer=optimizers.Adam(lr=LEARNING_RATE),
                  metrics=['acc'])

    if (i==0):
        model.summary()

    print('------------------------------------------------------------------------')
    print(f'Training for fold'+str(i+1))

    #print(len(train_batches))
    #print(len(valid_batches))

    STEP_SIZE_TRAIN = train_batches.n // train_batches.batch_size
    #STEP_SIZE_VALID = valid_batches.n // valid_batches.batch_size

    result = model.fit(train_batches,
                       steps_per_epoch=STEP_SIZE_TRAIN,
                       epochs=NUM_EPOCHS,
                       )

    # model.save()

    import matplotlib.pyplot as plt


    def plot_acc_loss(result, epochs):
        acc = result.history['acc']
        loss = result.history['loss']
        val_acc = result.history['val_acc']
        val_loss = result.history['val_loss']
        plt.figure(figsize=(15, 5))
        plt.subplot(121)
        plt.plot(range(1, epochs), acc[1:], label='Train_acc')
        plt.plot(range(1, epochs), val_acc[1:], label='Val_acc')
        plt.title('Accuracy over ' + str(epochs) + ' Epochs', size=12)
        plt.legend()
        plt.grid(True)
        plt.subplot(122)
        plt.plot(range(1, epochs), loss[1:], label='Train_loss')
        plt.plot(range(1, epochs), val_loss[1:], label='Val_loss')
        plt.title('Loss over ' + str(epochs) + ' Epochs', size=15)
        plt.legend()
        plt.grid(True)
        plt.show()


    # plot_acc_loss(result, NUM_EPOCHS)


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


    pred_generator.reset()

    count = [0, 0, 0]

    files = pred_generator.filenames

    for i in range(len(files)):
        x, y = pred_generator.next()
        img = x
        predict = model.predict(img)
        p = np.argmax(predict, axis=-1)
        p = model.predict_classes(img)
        count[p[0]] += 1


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
    print('------------------------------------------------------------------------')



# == Provide average scores ==
print('------------------------------------------------------------------------')
print('Score per fold')
for i in range(0, len(acc_list)):
  print('------------------------------------------------------------------------')
  print(f'> Fold {i+1} - Loss: {loss_list[i]} - Accuracy: {acc_list[i]}%')

print('------------------------------------------------------------------------')
print('Average scores for all folds:')
print(f'> Accuracy: {np.mean(acc_list)} (+- {np.std(acc_list)})')
print(f'> Loss: {np.mean(loss_list)}')
print('------------------------------------------------------------------------')