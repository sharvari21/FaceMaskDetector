import cv2 as cv
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os

print(cv.__version__)
# initialize the initial learning rate and epochs and batch size
ini_lr = 1e-4
epochs = 20
batch_size = 30

# assigning the dataset path to a variable and divide it into category

Dict_path = r"C:\Users\sharv\PycharmProjects\FaceMaskDetector\dataset"
categories = ["with_mask", "without_mask"]

# the list of data (i.e., images) and class images
print("[INFO] loading images...")
data = []
labels = []

# get the list of images from the dataset and initialize the data i.e. images

for cat in categories:
    path = os.path.join(str(Dict_path), str(cat))
    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        image = load_img(img_path, target_size=(250, 250))
        image = img_to_array(image)
        image = preprocess_input(image)

        data.append(image)
        labels.append(cat)

# Perform one-hot encoding on labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

data = np.array(data, dtype='float32')
labels = np.array(labels)

trainX, testX, trainY, testY = train_test_split(data, labels, test_size=0.20, stratify=labels, random_state=42)

# construct the training image generator for data augmentation

aug = ImageDataGenerator(rotation_range=20,
                         zoom_range=0.15,
                         width_shift_range=0.2,
                         height_shift_range=0.2,
                         shear_range=0.15,
                         horizontal_flip=True,
                         fill_mode='nearest')
# load the MobileNetV2 network, ensuring the head FC layer sets are
# left off
base_model = MobileNetV2(weights="imagenet",
                        include_top=False,
                        input_tensor=Input(shape=(250, 250, 3)))

# construct the head of the model that will be placed on top of the
# the base model
head_model = base_model.output
head_model =AveragePooling2D(pool_size=(7,7))(head_model)
head_model = Flatten(name='flatten')(head_model)
head_model = Dense(128, activation='relu')(head_model)
head_model = Dropout(0.5)(head_model)
head_model = Dense(2, activation='softmax')(head_model)

# place the head FC model on top of the base model (this will become
# the actual model we will train)
model = Model(inputs=base_model.input, outputs=head_model)

# loop over all layers in the base model and freeze them so they will
# *not* be updated during the first training process
for layer in base_model.layers:
    layer.trainable = False

# compile the model now
print('[INFO] Compiling Model..')
opt = Adam(lr=ini_lr, decay=ini_lr/epochs)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=["accuracy"])

# train the head of the network
print("[INFO] training head model..")
H_M = model.fit(aug.flow(trainX, trainY, batch_size=batch_size),
                steps_per_epoch=len(trainX) // batch_size,
                validation_data=(testX, testY),
                validation_steps=len(testX) // batch_size,
                epochs=epochs)
# Make prediction on testing test

print("[INFO] Evaluating network..")
pred = model.predict(testX, batch_size=batch_size)

# for each image in the testing set we need to find the index of the
# label with corresponding largest predicted probability
pred = np.argmax(pred, axis=1)

# show a nicely formatted classification report
print(classification_report(testY.argmax(axis=1), pred, target_names=lb.classes_))

# serialize the model to disk
print("[INFO] saving mask detector model..")
model.save("facemaskdetector.model", save_format="h5")

# plot the training accuracy and loss

n = epochs
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, n), H_M.history["loss"], label="train_loss")
plt.plot(np.arange(0, n), H_M.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, n), H_M.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, n), H_M.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")



