"""
ASL is the main form of communication for the Deaf and Hard-of-Hearing community in America. People with disabilities including Autism, Apraxia of speech, Cerebral Palsy, and Down Syndrome may also find this sign language beneficial for communicating.

In this python file, I will train a convolutional neural network to classify images of American Sign Language (ASL) letters.
I have used ASL Dataset from Kaggle.
Dataset link: 
https://www.kaggle.com/grassknoted/asl-alphabet 

"""

#Importing necessary libraries
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import os
import cv2


#Loading images to create train and test dataset
# setting up directories
train_dir = "/asl_alphabet_train/asl_alphabet_train"
test_dir = "/asl_alphabet_test/asl_alphabet_test"

#function to load train images from asl_alphabet_train folder.
#subfolder name is used to create labels for each image.
def load__train_images(directory):
    images = []
    labels = []
    for idx, label in enumerate(uniq_labels):
        for file in os.listdir(directory + "/" + label):
            filepath = directory + "/" + label + "/" + file
            image = cv2.resize(cv2.imread(filepath), (64, 64))
            images.append(image)
            labels.append(idx)
    images = np.array(images)
    labels = np.array(labels)
    return(images, labels)

#function to load test images from asl_alphabet_test folder
def load_test_images(directory):
    images = []
    labels = []
    for idx, label in enumerate(uniq_labels2):
        filepath = directory + "/" + label
        image = cv2.resize(cv2.imread(filepath), (64, 64))
        images.append(image)
        labels.append(idx)
    images = np.array(images)
    labels = np.array(labels)
    return(images, labels)

#function for printing one image of each class. [train and validation dataset]
def print_images(image_list):
    n = int(len(image_list) / len(uniq_labels))
    fig = plt.figure(figsize = (10, 20))

    for i in range(len(uniq_labels)):
        ax = plt.subplot(6, 5, i + 1)
        plt.imshow(image_list[int(n*i)])
        plt.title(uniq_labels[i])
        ax.title.set_fontsize(20)
        ax.axis('off')
    plt.show()

#print test images function
def print_test_images(image_list):
    n = int(len(image_list) / len(uniq_labels2))
    fig = plt.figure(figsize = (10, 20))

    for i in range(len(uniq_labels2)):
        ax = plt.subplot(6, 5, i + 1)
        plt.imshow(image_list[int(n*i)])
        plt.title(uniq_labels2[i])
        ax.title.set_fontsize(15)
        ax.axis('off')
    plt.show()

#train folder images
uniq_labels = sorted(os.listdir(train_dir))
images, labels = load__train_images(directory = train_dir)
print(uniq_labels)
print(np.unique(labels))

#test folder
uniq_labels2 = sorted(os.listdir(test_dir))
X_test, y_test = load_test_images(directory = test_dir)
print(uniq_labels2)
print(np.unique(y_test))

#splitting the images set into train and validation set.

from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size = 0.1, stratify = labels)

print("X_train shape: ",X_train.shape)
print("X_val shape: ",X_val.shape)
print("X_test shape: ",X_test.shape)


#printing training images
y_train_in = y_train.argsort()
y_train = y_train[y_train_in]
X_train = X_train[y_train_in]
print("Training Images: ")
print_images(X_train) #calling printing images function

#printing validation images
y_val_in = y_val.argsort()
y_val = y_val[y_val_in]
X_val = X_val[y_val_in]
print("validation images: ")
print_images(image_list = X_val) #calling printing images function


#printing test images
y_test_in = y_test.argsort()
y_test = y_test[y_test_in]
X_test = X_test[y_test_in]
print("Test images: ")
print_test_images(X_test)#calling printing images function

""" Preprocessing and Normalize RGB values """

y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)
y_val = keras.utils.to_categorical(y_val)

#scaling up each pixel value between -1 and 1
X_train = (X_train.astype('float32')/127.5)-1.0
X_test = (X_test.astype('float32')/127.5)-1.0
X_val = (X_val.astype('float32')/127.5)-1.0

""" Buliding classification model using CNN """

model=keras.models.Sequential([
    
    keras.layers.Conv2D(32,kernel_size=3,strides=1,padding='same',activation='relu',input_shape=(64,64,3)),
    keras.layers.Conv2D(32,kernel_size=3,strides=2,padding='same',activation='relu'), 

    keras.layers.Conv2D(64,kernel_size=3,strides=1,padding='same',activation='relu'),
    keras.layers.Conv2D(64,kernel_size=3,strides=2,padding='same',activation='relu'),   
    
    keras.layers.Conv2D(128,kernel_size=3,strides=1,padding='same',activation='relu'),
    keras.layers.Conv2D(128,kernel_size=3,strides=2,padding='same',activation='relu'),
    
    keras.layers.Flatten(),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(29,activation='softmax')    
    
])

model.summary()

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

history = model.fit(X_train, y_train, validation_data=(X_val,y_val),batch_size = 16, epochs=10)

""" Model Testing """

score = model.evaluate(x = X_train, y = y_train, verbose = 0)
print('Accuracy for train images:', round(score[1]*100, 3), '%')
score = model.evaluate(x = X_val, y = y_val, verbose = 0)
print('Accuracy for evaluation images:', round(score[1]*100, 3), '%')

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['train', 'validation'], loc='lower right')
plt.title('accuracy plot - train vs validation')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training loss', 'validation loss'], loc = 'upper right')
plt.title('loss plot - training vs vaidation')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

""" Predicting test images """

#predicting single image
print("Predicted output: ",np.argmax(model.predict(X_test[27][np.newaxis, :, :, 0:3])))
print("Actual output:",np.argmax(y_test[27]))
plt.imshow(X_test[27,:,:,0:3])

#For predicting all the test images.
# make predictions on an image and append it to the list (predictions).
predictions=[]
actual=[]
for i in range(len(uniq_labels2)):
  pred = np.argmax(model.predict(X_test[i][np.newaxis, :, :, 0:3]))
  if pred > 26:
    pred = pred - 1 #as del image is note there in test set
  act = np.argmax(y_test[i])
  predictions.append(uniq_labels2[pred])
  actual.append(uniq_labels2[act])

print(predictions)
print(actual)

fig = plt.figure(figsize = (24, 12))
for i in range(len(uniq_labels2)):
    ax = plt.subplot(8, 4, i+1)
    plt.subplots_adjust(hspace=1.5)
    plt.imshow(X_test[i,:,:,0:3])
    plt.title("Predicted: " + predictions[i] + "\nActual: " + actual[i])
    ax.title.set_fontsize(15)
    ax.axis('off')
plt.show()

