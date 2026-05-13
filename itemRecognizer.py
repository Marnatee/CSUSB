# Clothing item recognition for Fashion-MNIST dataset using Convolutional Neural Networks
# Classes in Fashion-MNIST:
# 0 = T-shirt/top, 1 = Trouser, 2 = Pullover, 3 = Dress, 4 = Coat
# 5 = Sandal, 6 = Shirt, 7 = Sneaker, 8 = Bag, 9 = Ankle boot

# Step 1: Import all required keras libraries
#Fashion dataset that will be trained and tested on
from keras.datasets import fashion_mnist
#Categorical format conversion
from tensorflow.keras.utils import to_categorical
#Used to build CNN layer by layer
from keras.models import Sequential
#CNN layers
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.layers import Dropout, BatchNormalization
#Adam optimizer to train the model
from keras.optimizers import Adam
#np.argmax() to find maximum value
import numpy as np

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Step 2: Load and return training and test datasets
#Function loads 60,000 training images and 10,000 testing images
#fashion_mnist is already split into 60,000 and 10,000 hardcoded into the dataset
def load_dataset():
	# 2a. Load dataset X_train, X_test, y_train, y_test via imported keras library
	(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

	# 2b. reshape for X train and test vars 
	# Hint: X_train = X_train.reshape((X_train.shape[0], 28, 28, 1)).astype('float32')
	X_train = X_train.reshape((X_train.shape[0], 28, 28, 1)).astype('float32')
	X_test = X_test.reshape((X_test.shape[0], 28, 28, 1)).astype('float32')

	# 2c. normalize inputs from 0-255 to 0-1 
	# Hint: X_train = X_train / 255
	X_train = X_train / 255
	X_test = X_test / 255

	# 2d. Convert y_train and y_test to categorical classes 
	# Hint: y_train = np_utils.to_categorical(y_train)
	y_train = to_categorical(y_train, 10)
	y_test = to_categorical(y_test, 10)

	# 2e. return your X_train, X_test, y_train, y_test
	return X_train, X_test, y_train, y_test

# Step 3: define your CNN model here in this function and then later use this function to create your model
def item_recognition_cnn():
	# 3a. create your CNN model here with Conv + ReLU + Flatten + Dense layers
	#Creates empty neural network
	model = Sequential()
	#Adds convolution layers that detect image features
	model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)))
	#Helps stabilize and speed up training
	model.add(BatchNormalization())
	model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
	#Shrinks feature maps while keeping important information
	model.add(MaxPooling2D((2, 2)))
	#Randomly disables underutilized neurons during training to reduce overfitting
	model.add(Dropout(0.25))
	model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
	model.add(BatchNormalization())
	model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
	model.add(MaxPooling2D((2, 2)))
	model.add(Dropout(0.25))
	#Converts 2D feature maps into a 1D vector
	model.add(Flatten())
	#Adds a fully-connected hidden layer
	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.50))
	#Outputs probabilities for the 10 clothing classes
	model.add(Dense(10, activation='softmax'))

	# 3b. Compile your model with categorical_crossentropy (loss), adam optimizer and accuracy as a metric
	model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

	# 3c. return your model
	return model

def run_training():
	#Loads and prepares the dataset
	X_train, X_test, y_train, y_test = load_dataset()
	# Step 4: Call item_recognition_cnn() to build your model
	model = item_recognition_cnn()
	#Prints the model structure
	model.summary()

	# Step 5: Train your model and see the result in Command window.
	# Set epochs to a number between 10 - 20 and batch_size between 150 - 200
	history = model.fit(X_train, y_train, epochs=10, batch_size=180, validation_split=0.10, verbose=1)

	# Step 6: Evaluate your model via your_model_name.evaluate() function and copy the result in your report
	#Tests the model on the official test dataset
	score = model.evaluate(X_test, y_test, verbose=0)
	#Test loss is how confident the predictions are. Lower number = better.
	print('Test loss:', score[0])
	#Test accuracy is how accurate the predictions were. Higher number = better.
	print('Test accuracy:', score[1])

	#Predicts the first 10 test images
	predictions = model.predict(X_test[:10])
	print('Predicted classes for first 10 test images:', np.argmax(predictions, axis=1))
	print('Actual classes for first 10 test images:', np.argmax(y_test[:10], axis=1))

	# Step 7: Save your model via your_model_name.save('itemRecognizer.h5')
	model.save('itemRecognizer.h5')
	print('Saved model as itemRecognizer.h5')

	# Step 7a (recommended): Use these plots for training/validation accuracy and loss in your report

	import matplotlib.pyplot as plt

	#Creates and saves accuracy/loss graphs
	plt.figure(figsize=(8, 5))
	plt.plot(history.history['accuracy'], label='Training Accuracy')
	plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
	plt.xlabel('Epoch')
	plt.ylabel('Accuracy')
	plt.title('Training vs Validation Accuracy')
	plt.legend()
	plt.savefig('training_validation_accuracy.png')
	plt.close()

	plt.figure(figsize=(8, 5))
	plt.plot(history.history['loss'], label='Training Loss')
	plt.plot(history.history['val_loss'], label='Validation Loss')
	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	plt.title('Training vs Validation Loss')
	plt.legend()
	plt.savefig('training_validation_loss.png')
	plt.close()
	print('Saved plots as training_validation_accuracy.png and training_validation_loss.png')



# Code below to make a prediction for a new image.

# Step 8: load required keras libraries
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model
 
# Step 9: load and normalize new image
#Function loads one custom image as grayscale and resizes it to 28x28
def load_new_image(path):
	# 9a. load new image (grayscale clothing image)
	newImage = load_img(path, color_mode='grayscale', target_size=(28, 28))
	# 9b. Convert image to array
	newImage = img_to_array(newImage)
	# 9c. reshape into a single sample with 1 channel 
	# Hint: newImage = newImage.reshape(1, 28, 28, 1)
	newImage = newImage.reshape(1, 28, 28, 1).astype('float32')
	
	# 9d. normalize image data 
	# Hint: newImage = newImage / 255
	newImage = newImage / 255
	
	# 9e. return newImage
	return newImage

# Step 10: load a new image and predict its class
def test_model_performance():
	# 10a. Call the above load image function
	# img = load_new_image('sample_images/sneaker.png')
	# List of sample images
	image_files = [
		'tshirt_top.png', 'trouser.png', 'pullover.png', 'dress.png', 'coat.png',
		'sandal.png', 'shirt.png', 'sneaker.png', 'bag.png', 'ankle_boot.png'
	]
	# 10b. load your CNN model (itemRecognizer.h5 file)
	your_model_name = load_model('itemRecognizer.h5')

	#Use every sample image in the folder and print a prediction for them
	for image_file in image_files:
		path = 'sample_images/' + image_file

		print('\nTesting:', image_file)

		# Load and preprocess image
		img = load_new_image(path)

		# 10c. predict the class
		# Hint: imageClass = your_model_name.predict(img)
		imageClass = your_model_name.predict(img)
		#pick the highest value probability stored in imageClass
		class_idx = int(np.argmax(imageClass[0]))
		# 10d. Print prediction result
		print(imageClass[0])
		print('Predicted class index:', class_idx)
		print('Predicted label:', class_names[class_idx])

# Step 11: Test model performance here by calling the above test_model_performance function
run_training()
test_model_performance()