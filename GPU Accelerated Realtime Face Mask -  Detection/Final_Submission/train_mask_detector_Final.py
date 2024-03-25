# importing libraries

import os # To interact with operating system
import numpy as np # to work with arrays
import matplotlib.pyplot as plt # To create a plot
from imutils import paths # To traverse directories
from tensorflow.keras.preprocessing.image import ImageDataGenerator # To generate more images by doing data augmentation
from tensorflow.keras.optimizers import legacy as keras_legacy # importing optimizer from keras.legacy
from tensorflow.keras.applications import MobileNetV2 # importing MobileNetV2
from tensorflow.keras.layers import AveragePooling2D # To apply Average Pooling for 2D spatial Data
from tensorflow.keras.layers import Dropout # To prevent overfitting
from tensorflow.keras.layers import Flatten # To Flattern the input
from tensorflow.keras.layers import Dense # Importing Dense Layer
from tensorflow.keras.layers import Input # Importing input layer
from tensorflow.keras.models import Model # To create the final model
from tensorflow.keras.optimizers import Adam # Importing adam optimizer
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input # To preprocess the input
from tensorflow.keras.preprocessing.image import img_to_array # To convert PIL image to numpy array
from tensorflow.keras.preprocessing.image import load_img # To load an image from file
from tensorflow.keras.utils import to_categorical # To convert class vectors to binary class matrices
from sklearn.preprocessing import LabelBinarizer # To convert categorical labels to binary labels
from sklearn.model_selection import train_test_split # To split arrays or matrices into random train and test subsets
from sklearn.metrics import classification_report # To generate a text report showing the main claassification metrics
from sklearn.metrics import confusion_matrix
import seaborn as sns 

# initializing hyperparamenters(leraning rate, number of epochs and batch size)
Learning_Rate = 1e-4
EPOCHS = 20
Batch_Size = 32

# Defining the path of the dataset
file_dir = r"C:\Users\naidu\OneDrive\Documents\ML PROJECT\Final_Submission\Face Mask Dataset"
#Defining the classes of the dataset
n_classes = ["with_mask", "without_mask"]

print("loading images...")

data = [] # To store images
labels = [] # to store the corresponding labels

for category in n_classes: # Iterating through each class
	path = os.path.join(file_dir, category) # Getting path  of the class
	for img in os.listdir(path): # Looping through each image in the class
		img_path = os.path.join(path, img) # Getting path of the image
		image = load_img(img_path, target_size=(224, 224)) # Loading the image and resizing it to 224 x 224
		image = img_to_array(image) #PIL image to numpy array 
		image = preprocess_input(image) 
		data.append(image)
		labels.append(category)

# one-hot encoding on the labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

# Convert the data and labels to numpy arrays
data = np.array(data, dtype="float32")
labels = np.array(labels)

# Split the data into training and testing sets
(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=0.20, stratify=labels, random_state=42)

# construct an image generator for data augmentation during training
data_gen = ImageDataGenerator(
	rotation_range=20,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")

# load MobileNetV2 network with pre-trained weights on ImageNet dataset
# Exclude the head fully connected (FC) layers in the network
baseModel = MobileNetV2(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(224, 224, 3)))

# HeadModel will be placed on top of the base model
headModel = baseModel.output
# Applying averaage pooling on the output features maps from base model
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
# Flatten the pooled features
headModel = Flatten(name="flatten")(headModel)
# fully connected layer with 128 neurons and ReLU as activation
headModel = Dense(128, activation="relu")(headModel)
#Dropout layer is to prevent overfitting
headModel = Dropout(0.5)(headModel)
# fully connected layer with 2 neurons and softmax as activation
headModel = Dense(2, activation="softmax")(headModel)

# Create the final model by placing the head fully connected layer on top of the base MobileNetV2 model
model = Model(inputs=baseModel.input, outputs=headModel)

# Freeze all layers in the base MobileNetV2 model to prevent their weights from being updated during the first training process
for layer in baseModel.layers:
	layer.trainable = False

# Compiling our model
print("Compiling Model...")
model.compile(loss="binary_crossentropy", optimizer= keras_legacy.Adam(learning_rate=Learning_Rate, decay=Learning_Rate / EPOCHS),
	metrics=["accuracy"])

# Train the head of the network using the data generator
print("Training head...")
Head_train = model.fit(
	data_gen.flow(trainX, trainY, batch_size=Batch_Size),
	steps_per_epoch=len(trainX) // Batch_Size,
	validation_data=(testX, testY),
	validation_steps=len(testX) // Batch_Size,
	epochs=EPOCHS)

# Plot the training loss and accuracy
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), Head_train.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), Head_train.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), Head_train.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), Head_train.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")

# Make predictions on the testing set
print("[Evaluating Network...")
pred_test_set = model.predict(testX, batch_size=Batch_Size)

# For each image in the testing set, find the index of the label with corresponding largest predicted probability
pred_test_set = np.argmax(pred_test_set, axis=1)

# Classification Report
print(classification_report(testY.argmax(axis=1), pred_test_set,
	target_names=lb.classes_))

# Get the model's predictions on the test set
preds = model.predict(testX, batch_size=Batch_Size)

# Convert the predictions and true labels to class values (i.e. 0 or 1)
pred_classes = np.argmax(preds, axis=1)
true_classes = np.argmax(testY, axis=1)

# Compute the confusion matrix
conf_matrix = confusion_matrix(true_classes, pred_classes)

# Plot the confusion matrix as a heatmap
sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.show()


# Saving the trained model
print("Saving Mask Detector Model...")
model.save("facemask_detector_MobileNetV2.model", save_format="h5")
