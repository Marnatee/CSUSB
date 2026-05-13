# Clothing item recognition for Fashion-MNIST dataset using Convolutional Neural Networks

# Step 1: Import all required keras libraries
from keras.models import load_model
# Fashion dataset that will be trained and tested on
from keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical
# np.argmax() to find maximum value
import numpy as np

# NEW: imports for reading all files in sample_images/
import os

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Step 2: Load and return training and test datasets
def load_dataset():
	# 2a. Load dataset X_train, X_test, y_train, y_test via imported keras library
	(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

	# 2b. reshape for X train and test vars
	X_train = X_train.reshape((X_train.shape[0], 28, 28, 1)).astype('float32')
	X_test = X_test.reshape((X_test.shape[0], 28, 28, 1)).astype('float32')

	# 2c. normalize inputs from 0-255 to 0-1
	X_train = X_train / 255
	X_test = X_test / 255

	# 2d. Convert y_train and y_test to categorical classes
	y_train = to_categorical(y_train, 10)
	y_test = to_categorical(y_test, 10)

	# 2e. return your X_train, X_test, y_train, y_test
	return X_train, X_test, y_train, y_test

# Step 3: Load your saved model
model = load_model('itemRecognizer.h5')

# Step 4: Evaluate your model
X_train, X_test, y_train, y_test = load_dataset()

score = model.evaluate(X_test, y_test, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Code below to make a prediction for a new image.
predictions = model.predict(X_test[:10])
print('Predicted classes for first 10 test images:', np.argmax(predictions, axis=1))
print('Actual classes for first 10 test images:', np.argmax(y_test[:10], axis=1))

# Step 5: load required keras libraries
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

# Step 6: load and normalize new image
def load_new_image(path):
	# 6a. load new image
	newImage = load_img(path, color_mode='grayscale', target_size=(28, 28))

	# 6b. Convert image to array
	newImage = img_to_array(newImage)

	# SAVE ORIGINAL PIXELS BEFORE NORMALIZATION
	original_pixels = newImage.copy()

	# 6c. reshape into a single sample with 1 channel
	newImage = newImage.reshape(1, 28, 28, 1).astype('float32')

	# 6d. normalize image data
	newImage = newImage / 255

	# 6e. return normalized image + original pixel values
	return newImage, original_pixels

# NEW FUNCTION: Print 28x28 image pixels to terminal
def print_image_pixels(pixel_array):
	print('\n28x28 Pixel Values:\n')

	# Remove channel dimension (28,28,1 -> 28,28)
	pixel_array = pixel_array.squeeze()

	for row in pixel_array:
		for pixel in row:
			print(f'{int(pixel):3}', end=' ')
		print()

# Step 7: load every image in sample_images and predict class
def test_model_performance():

	# Load CNN model
	your_model_name = load_model('itemRecognizer.h5')

	# Folder containing images
	sample_folder = 'sample_images'

	# Automatically grab every image file in sample_images/
	image_files = [
		f for f in os.listdir(sample_folder)
		if f.lower().endswith(('.png', '.jpg', '.jpeg'))
	]

	# Sort files alphabetically
	image_files.sort()

	# Use every sample image in the folder
	for image_file in image_files:

		path = os.path.join(sample_folder, image_file)

		print('\n' + '=' * 60)
		print('FILE NAME:', image_file)
		print('=' * 60)

		# Load and preprocess image
		img, original_pixels = load_new_image(path)

		# Print the 28x28 pixel image to terminal
		print_image_pixels(original_pixels)

		# Predict the class
		imageClass = your_model_name.predict(img)

		# Pick highest probability
		class_idx = int(np.argmax(imageClass[0]))

		# Print prediction result
		print('\nPrediction Probabilities:')
		print(imageClass[0])

		print('\nPredicted class index:', class_idx)
		print('Predicted label:', class_names[class_idx])
		print()

# Find incorrect predictions
predictions = model.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)
actual_classes = np.argmax(y_test, axis=1)

wrong_indexes = np.where(predicted_classes != actual_classes)[0]

print('Number of wrong predictions:', len(wrong_indexes))
print('First 10 wrong indexes:', wrong_indexes[:10])

for i in wrong_indexes[:10]:
	print('Image index:', i)
	print('Predicted:', predicted_classes[i], class_names[predicted_classes[i]])
	print('Actual:', actual_classes[i], class_names[actual_classes[i]])
	print()

# Run predictions on all images inside sample_images/
test_model_performance()