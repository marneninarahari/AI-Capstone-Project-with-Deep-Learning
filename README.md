# IBM-AI-Capstone-Project-with-Deep-Learning

### AI Project with Deep Learning: Image Classification

### Project Overview:

•	The dataset used is concrete crack images for classification, there are 40,000 color images, 20,000 with cracks (positive) and 20,000 with no cracks (negatives). 75% of the images for training, 23.75% for validation, and 1.25% for testing are used.

#### I. Image Classification Using Keras

•	Built an image classifier using the VGG16 pre-trained model, evaluated and compared its performance with another model built by using the ResNet50 pre-trained model.

•	Imported libraries, modules, and packages needed. Importantly the preprocess_input function from keras applications and used a batch size of 100 images for both training and validation.

•	Constructed an ImageDataGenerator for the training set and another one for the validation set.

•	Compiled the model using the Adam optimizer and the categorical_crossentropy loss function.

•	The performances of the classifier using the VGG16 pre-trained model and the classifier using the ResNet50 pre-trained model were compared.

•	It was observed that the performance of VGG16 trained model (Loss = 0.007938 and Accuracy: 0.998) is slightly lower than the performance of ResNet50 trained model on test data (Loss = 0.00238 and Accuracy = 1.0).

#### II. Image Classification Using PyTorch

•	Created the dataset object class. Then, two dataset objects, one for the training data and one for the validation data were generated.

•	The pre-trained resnet18 model was used and the parameter ‘pretrained’ was set to true. Then, the attribute ‘requires_grad’ was set to false so that the parameters will not be affected by training.

•	The model was trained by creating a training loader and validation loader object with the batch size of 100 samples each. CrossEntropyLoss function and Adam optimizer were used.

•	The model was validated on test data and it was found that the model accuracy is 0.9943 for 300 iterations. Also, first four misclassified images were identified using the validation data set.
