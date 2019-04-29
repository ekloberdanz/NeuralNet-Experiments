In this lab I experimented with Neural Network (NN)classifiers for handwritten digit optical recognition (a multi-class problem) using
Keras with TensorFlow.
I created 3 groups of fully connected feed-forward (FNN) and 1 class of Convolutional Neural Net (CNN). All NNs use the softmax function
for the output layer, which converts numbers (logits) into probabilities that sum up to one. Additionally, all NNs use 20% of training
data as the validation set, 1-of-c output encoding (keras.utils.to_categorical) and early stopping to decide when to stop training to
prevent overfitting (keras.callbacks.EarlyStopping). Also, I created with 20 different models for every FNN (60 in total) and 10 different
models for the CNN and recorded the best model for the FNNs and the CNN including the overall classification accuracy, class accuracy, and
confusion matrix for both training and testing data.

All 4 programs mse_relu.py (Task 1.a), cross_entropy_relu.py (Task 1.a), cross_entropy_tanh.py (Task 1. b), and cnn_relu.py (Task 2) must be run with python3 (it was made with and tested with Python 3.5.2 on ubuntu).

Install dependencies: pip install -r requirements.txt 

usage:type into terminal: 
python3 mse_relu.py training.csv testing.csv
python3 cross_entropy_relu.py training.csv testing.csv
python3 cross_entropy_tanh.py training.csv testing.csv
python3 cnn_relu.py training.csv testing.csv

(Each of the first three programs trains 20 models and the last one trains 10 models - takes about a minute to run a program.)

the csv files that contain the data are provided in the submission folder
