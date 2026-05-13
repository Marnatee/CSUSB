Take 70,000 images from Fashion-MNST and use 60,000 to train a CNN model and 10,000 to test it.

run itemRecognizer.py to train the model
creates:
itemRecognizer.h5
training_validation_accuracy.png
training_validation_loss.png

run evaluation.py to test the model
(requires itemRecognizer.h5 to run so generate with itemRecognizer.py or use included file)
will use any files in the sample_images folder
(must be 28x28 grayscale images)
