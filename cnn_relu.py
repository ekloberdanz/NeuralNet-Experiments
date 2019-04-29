import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from sklearn.metrics import confusion_matrix
import time
from keras.optimizers import *
from matplotlib.pyplot import *
import matplotlib.pyplot as plt
import numpy
import inspect
import sys
from tabulate import tabulate

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)


# Model
def createNN_1layer(filters, kernel_size, strides, padding, output_dim, activation, loss, weights, bias, learning_rate, optimizer_type, momentum=None):
	classifier = Sequential()
	#First Hidden Layer
	classifier.add(Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, activation=activation, kernel_initializer=weights, bias_initializer=bias, input_shape=(8,8,1)))
	classifier.add(MaxPooling2D(2, 2))
	classifier.add(Flatten())
	#Output Layer (10 digits => 10 neurons)
	classifier.add(Dense(output_dim, activation='softmax', kernel_initializer=weights, bias_initializer=bias))
	#Compiling the neural network
	if optimizer_type == SGD:
		optimizer = optimizer_type(lr=learning_rate, momentum=momentum)
	else:
		optimizer = optimizer_type(lr=learning_rate) # E.G.: Adam and Adamax don't have parameter momemntum
	classifier.compile(optimizer = optimizer,loss=loss, metrics =['accuracy'])
	return(classifier, filters, kernel_size, strides, padding, output_dim, activation, loss, weights, bias, learning_rate, optimizer_type, momentum)


def createNN_2layer(filters, kernel_size, strides, padding, output_dim, activation, loss, weights, bias, learning_rate, optimizer_type, momentum=None):
	classifier = Sequential()
	#First Hidden Layer
	classifier.add(Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, activation=activation, kernel_initializer=weights, bias_initializer=bias, input_shape=(8,8,1)))
	#classifier.add(MaxPooling2D(2, 2))
	# Second hidden layer
	classifier.add(Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, activation=activation, kernel_initializer=weights, bias_initializer=bias, input_shape=(8,8,1)))
	classifier.add(MaxPooling2D(2, 2))
	classifier.add(Flatten())
	#Output Layer (10 digits => 10 neurons)
	classifier.add(Dense(output_dim, activation='softmax', kernel_initializer=weights, bias_initializer=bias))
	#Compiling the neural network
	if optimizer_type == SGD:
		optimizer = optimizer_type(lr=learning_rate, momentum=momentum)
	else:
		optimizer = optimizer_type(lr=learning_rate) # E.G.: Adam and Adamax don't have parameter momemntum
	classifier.compile(optimizer = optimizer,loss=loss, metrics =['accuracy'])
	return(classifier, filters, kernel_size, strides, padding, output_dim, activation, loss, weights, bias, learning_rate, optimizer_type, momentum)


def createNN_3layer(filters, kernel_size, strides, padding, output_dim, activation, loss, weights, bias, learning_rate, optimizer_type, momentum=None):
	classifier = Sequential()
	#First Hidden Layer
	classifier.add(Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, activation=activation, kernel_initializer=weights, bias_initializer=bias, input_shape=(8,8,1)))
	#classifier.add(MaxPooling2D(2, 2))
	# Second hidden layer
	classifier.add(Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, activation=activation, kernel_initializer=weights, bias_initializer=bias, input_shape=(8,8,1)))
	#classifier.add(MaxPooling2D(2, 2))
	# Third hidden layer
	classifier.add(Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, activation=activation, kernel_initializer=weights, bias_initializer=bias, input_shape=(8,8,1)))
	classifier.add(MaxPooling2D(2, 2))
	classifier.add(Flatten())
	#Output Layer (10 digits => 10 neurons)
	classifier.add(Dense(output_dim, activation='softmax', kernel_initializer=weights, bias_initializer=bias))
	#Compiling the neural network
	if optimizer_type == SGD:
		optimizer = optimizer_type(lr=learning_rate, momentum=momentum)
	else:
		optimizer = optimizer_type(lr=learning_rate) # E.G.: Adam and Adamax don't have parameter momemntum
	classifier.compile(optimizer = optimizer,loss=loss, metrics =['accuracy'])
	return(classifier, filters, kernel_size, strides, padding, output_dim, activation, loss, weights, bias, learning_rate, optimizer_type, momentum)


results = []

# Load training data and labels
train = sys.argv[1]
train = pd.read_csv(train,header=None)
# Load testing data and labels
test = sys.argv[2]
test = pd.read_csv(test,header=None)
# Reshape training data
X_train = train.iloc[:, 0:64].values.reshape(train.shape[0], 8,8,1)
#print("after", X_train.shape)

# Training labels
y_train = train.iloc[:,64].values

# Encode training labels to 1-of-c output
y_train = keras.utils.to_categorical(y_train,num_classes=10)

# Convert data to a numpy array (required for Keras)
X_train = np.array(X_train)
y_train = np.array(y_train)

# Use 20% of training dataset as validation data
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size = 0.2)

# Reshape testing data, define a variable for labels, and convert to numpy array
X_test = test.iloc[:, 0:64].values.reshape(test.shape[0], 8,8,1)
y_test = test.iloc[:,64].values
y_test = keras.utils.to_categorical(y_test,num_classes=10)
X_test = np.array(X_test)
y_test = np.array(y_test)

classifier1 = createNN_1layer(filters=64, kernel_size = (5, 5), strides=(1,1), padding='same', output_dim=10, activation='relu', loss='categorical_crossentropy', weights='random_normal', bias='ones',	learning_rate = 0.1,  optimizer_type=SGD, momentum = 0.8)[0]
model1 = createNN_1layer(filters=64, kernel_size = (5, 5), strides=(1,1), padding='same', output_dim=10, activation='relu', loss='categorical_crossentropy', weights='random_normal', bias='ones', learning_rate = 0.1, optimizer_type=SGD, momentum = 0.8)[1:]

classifier2 = createNN_1layer(filters=64, kernel_size = (5, 5), strides=(1,1), padding='valid', output_dim=10, activation='relu', loss='categorical_crossentropy', weights='random_normal', bias='zeros',	learning_rate = 0.001, optimizer_type=Adam)[0]
model2 = createNN_1layer(filters=64, kernel_size = (5, 5), strides=(1,1), padding='valid', output_dim=10, activation='relu', loss='categorical_crossentropy', weights='random_normal', bias='zeros', learning_rate = 0.001, optimizer_type=Adam)[1:]

classifier3 = createNN_1layer(filters=64, kernel_size = (3, 3), strides=(1,1), padding='valid', output_dim=10, activation='relu', loss='categorical_crossentropy', weights='random_normal', bias='zeros',	learning_rate = 0.02, optimizer_type=SGD, momentum = 0.9)[0]
model3 = createNN_1layer(filters=64, kernel_size = (3, 3), strides=(1,1), padding='valid', output_dim=10, activation='relu', loss='categorical_crossentropy', weights='random_normal', bias='zeros', learning_rate = 0.02, optimizer_type=SGD, momentum = 0.9)[1:]

classifier4 = createNN_2layer(filters=30, kernel_size = (5, 5), strides=(2,2), padding='same', output_dim=10, activation='relu', loss='categorical_crossentropy', weights='random_normal', bias='zeros',	learning_rate = 0.01, optimizer_type=Adam)[0]
model4 = createNN_2layer(filters=30, kernel_size = (5, 5), strides=(1,1), padding='same', output_dim=10, activation='relu', loss='categorical_crossentropy', weights='random_normal', bias='zeros', learning_rate = 0.01, optimizer_type=Adam)[1:]

classifier5 = createNN_2layer(filters=10, kernel_size = (3, 3), strides=(2,2), padding='same', output_dim=10, activation='relu', loss='categorical_crossentropy', weights='random_uniform', bias='zeros',	learning_rate = 0.0001, optimizer_type=Adam)[0]
model5 = createNN_2layer(filters=10, kernel_size = (3, 3), strides=(2,2), padding='same', output_dim=10, activation='relu', loss='categorical_crossentropy', weights='random_uniform', bias='zeros', learning_rate = 0.0001, optimizer_type=Adam)[1:]

classifier6 = createNN_2layer(filters=64, kernel_size = (5, 5), strides=(1,1), padding='same', output_dim=10, activation='relu', loss='categorical_crossentropy', weights='random_normal', bias='ones',	learning_rate = 0.03, optimizer_type=Adam)[0]
model6 = createNN_2layer(filters=64, kernel_size = (5, 5), strides=(1,1), padding='same', output_dim=10, activation='relu', loss='categorical_crossentropy', weights='random_normal', bias='ones', learning_rate = 0.03, optimizer_type=Adam)[1:]

classifier7 = createNN_2layer(filters=15, kernel_size = (5, 5), strides=(1,1), padding='same', output_dim=10, activation='relu', loss='categorical_crossentropy', weights='random_normal', bias='zeros',	learning_rate = 0.001, optimizer_type=Adamax)[0]
model7 = createNN_2layer(filters=15, kernel_size = (5, 5), strides=(1,1), padding='same', output_dim=10, activation='relu', loss='categorical_crossentropy', weights='random_normal', bias='zeros', learning_rate = 0.001, optimizer_type=Adamax)[1:]

classifier8 = createNN_3layer(filters=5, kernel_size = (3, 3), strides=(1,1), padding='same', output_dim=10, activation='relu', loss='categorical_crossentropy', weights='random_normal', bias='zeros',	learning_rate = 0.001, optimizer_type=SGD, momentum = 0.95)[0]
model8 = createNN_3layer(filters=5, kernel_size = (3, 3), strides=(1,1), padding='same', output_dim=10, activation='relu', loss='categorical_crossentropy', weights='random_normal', bias='zeros', learning_rate = 0.001, optimizer_type=SGD, momentum=0.95)[1:]

classifier9 = createNN_3layer(filters=64, kernel_size = (5, 5), strides=(1,1), padding='same', output_dim=10, activation='relu', loss='categorical_crossentropy', weights='random_normal', bias='zeros',	learning_rate = 0.015, optimizer_type=SGD, momentum=0.7)[0]
model9 = createNN_3layer(filters=64, kernel_size = (5, 5), strides=(1,1), padding='same', output_dim=10, activation='relu', loss='categorical_crossentropy', weights='random_normal', bias='zeros', learning_rate = 0.015, optimizer_type=SGD, momentum=0.7)[1:]

classifier10 = createNN_3layer(filters=64, kernel_size = (3, 3), strides=(1,1), padding='same', output_dim=10, activation='relu', loss='categorical_crossentropy', weights='random_normal', bias='zeros',learning_rate = 0.01, optimizer_type=SGD, momentum=0.9)[0]
model10 = createNN_3layer(filters=64, kernel_size = (3, 3), strides=(1,1), padding='same', output_dim=10, activation='relu', loss='categorical_crossentropy', weights='random_normal', bias='zeros', learning_rate = 0.01, optimizer_type=SGD, momentum=0.9)[1:]

# Set up Early Stopping of training
callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss',min_delta=0,patience=2,verbose=0,mode='auto')]

print("\nInitiating training of 10 different models\n")

# model1
# Start time of training
start_time1=time.time()
# Trainng
batch_size1 = 10
epochs1 = 10
classifier1.fit(X_train,y_train, validation_data=(X_valid, y_valid), batch_size=batch_size1, epochs=epochs1,  callbacks=callbacks)
# Execution time
execution_time1 = time.time() - start_time1
# Evaluate (train data): losss and accuracy
eval_model1=classifier1.evaluate(X_train, y_train)
eval_model1_test=classifier1.evaluate(X_test, y_test)
results.append([1, batch_size1, epochs1, model1[0], model1[1], model1[2], model1[3], model1[4], model1[5], model1[6], model1[7], model1[8], model1[9],'SGD', model1[11], execution_time1, eval_model1[0], eval_model1[1], eval_model1_test[0], eval_model1_test[1]])


# model2
# Start time of training
start_time2=time.time()
# Trainng
batch_size2 = 10
epochs2 = 10
classifier2.fit(X_train,y_train, validation_data=(X_valid, y_valid), batch_size=batch_size2, epochs=epochs2,  callbacks=callbacks)
# Execution time
execution_time2 = time.time() - start_time2
# Evaluate (train data): losss and accuracy
eval_model2=classifier2.evaluate(X_train, y_train)
x=model2
eval_model2_test=classifier2.evaluate(X_test, y_test)
results.append([1, batch_size2, epochs2, x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9],'Adam',x[11], execution_time2, eval_model2[0], eval_model2[1], eval_model2_test[0], eval_model2_test[1]])

# model3
start_time3=time.time()
batch_size3 = 5
epochs3 = 15
classifier3.fit(X_train,y_train, validation_data=(X_valid, y_valid), batch_size=batch_size3, epochs=epochs3,  callbacks=callbacks)
execution_time3 = time.time() - start_time3# Execution time
eval_model3=classifier3.evaluate(X_train, y_train)# Evaluate (train data): losss and accuracy
x = model3
eval_model3_test=classifier3.evaluate(X_test, y_test)
results.append([1, batch_size3, epochs3, x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9],'SGD',x[11], execution_time3, eval_model3[0], eval_model3[1], eval_model3_test[0], eval_model3_test[1]])

# model4
start_time4=time.time()
batch_size4 = 10
epochs4 = 50
classifier4.fit(X_train,y_train, validation_data=(X_valid, y_valid), batch_size=batch_size4, epochs=epochs4,  callbacks=callbacks)
execution_time4 = time.time() - start_time4# Execution time
eval_model4=classifier4.evaluate(X_train, y_train)# Evaluate (train data): losss and accuracy
eval_model4_test=classifier4.evaluate(X_test, y_test)
x = model4
results.append([2, batch_size4, epochs4, x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9],'Adam',x[11], execution_time4, eval_model4[0], eval_model4[1], eval_model4_test[0], eval_model4_test[1]])

# model5
start_time5=time.time()
batch_size5 = 20
epochs5 = 20
classifier5.fit(X_train,y_train, validation_data=(X_valid, y_valid), batch_size=batch_size5, epochs=epochs5,  callbacks=callbacks)
execution_time5 = time.time() - start_time5# Execution time
eval_model5=classifier5.evaluate(X_train, y_train)# Evaluate (train data): losss and accuracy
eval_model5_test=classifier5.evaluate(X_test, y_test)
x = model5
results.append([2, batch_size5, epochs5, x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9],'Adam',x[11], execution_time5, eval_model5[0], eval_model5[1], eval_model5_test[0], eval_model5_test[1]])

# model6
start_time6=time.time()
batch_size6 = 10
epochs6 = 10
classifier6.fit(X_train,y_train, validation_data=(X_valid, y_valid), batch_size=batch_size6, epochs=epochs6,  callbacks=callbacks)
execution_time6 = time.time() - start_time6# Execution time
eval_model6=classifier6.evaluate(X_train, y_train)# Evaluate (train data): losss and accuracy
eval_model6_test=classifier6.evaluate(X_test, y_test)
x = model6
results.append([2, batch_size6, epochs6, x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9],'Adam',x[11], execution_time6, eval_model6[0], eval_model6[1], eval_model6_test[0], eval_model6_test[1]])

# model7
start_time7=time.time()
batch_size7 = 10
epochs7 = 10
classifier7.fit(X_train,y_train, validation_data=(X_valid, y_valid), batch_size=batch_size7, epochs=epochs7,  callbacks=callbacks)
execution_time7 = time.time() - start_time7# Execution time
eval_model7=classifier7.evaluate(X_train, y_train)# Evaluate (train data): losss and accuracy
eval_model7_test=classifier7.evaluate(X_test, y_test)
x = model7
results.append([2, batch_size7, epochs7, x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9],'Adamax',x[11], execution_time7, eval_model7[0], eval_model7[1], eval_model7_test[0], eval_model7_test[1]])

# model8
start_time8=time.time()
batch_size8 = 10
epochs8 = 5
classifier8.fit(X_train,y_train, validation_data=(X_valid, y_valid), batch_size=batch_size8, epochs=epochs8,  callbacks=callbacks)
execution_time8 = time.time() - start_time8# Execution time
eval_model8=classifier8.evaluate(X_train, y_train)# Evaluate (train data): losss and accuracy
eval_model8_test=classifier8.evaluate(X_test, y_test)
x = model8
results.append([3, batch_size8, epochs8, x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9],'SGD',x[11], execution_time8, eval_model8[0], eval_model8[1], eval_model8_test[0], eval_model8_test[1]])

# model9
start_time9=time.time()
batch_size9 = 15
epochs9 = 10
classifier9.fit(X_train,y_train, validation_data=(X_valid, y_valid), batch_size=batch_size9, epochs=epochs9,  callbacks=callbacks)
execution_time9 = time.time() - start_time9# Execution time
eval_model9=classifier9.evaluate(X_train, y_train)# Evaluate (train data): losss and accuracy
eval_model9_test=classifier9.evaluate(X_test, y_test)
x = model9
results.append([3, batch_size9, epochs9, x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9],'SGD',x[11], execution_time9, eval_model9[0], eval_model9[1], eval_model9_test[0], eval_model9_test[1]])

# model10
start_time10=time.time()
batch_size10 = 10
epochs10 = 10
classifier10.fit(X_train,y_train, validation_data=(X_valid, y_valid), batch_size=batch_size10, epochs=epochs10,  callbacks=callbacks)
execution_time10 = time.time() - start_time10# Execution time
eval_model10=classifier10.evaluate(X_train, y_train)# Evaluate (train data): losss and accuracy
eval_model10_test=classifier10.evaluate(X_test, y_test)
x = model10
results.append([3, batch_size10, epochs10, x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9],'SGD',x[11], execution_time10, eval_model10[0], eval_model10[1], eval_model10_test[0], eval_model10_test[1]])

# output results
results_df = pd.DataFrame(results, columns=('L', 'B', 'E', 'F', 'K', 'S', 'P', 'OD', 'activ', 'loss', 'weights', 'bias','lr',  'optim', 'momentum', 'exe_time', 'loss_train', 'acc_train', 'loss_test', 'acc_test'))
print("\nTable I: Results of sample experiments with cross entropy loss function and relu activation function\n")
print("legend: L = number of hidden layers, B = batch size, E = number of epochs, F = filters, K = kernel ,S = strides, P = padding, OD = output dimension\n")
print(tabulate(results_df, headers='keys'))

'''
# Graphs of loss and accuracy rate
plt.plot(model.history['loss'])
plt.plot(model.history['val_loss'])
plt.title('Loss Rate', size=14)
plt.ylabel('Loss')
plt.xlabel('Training interations')
plt.legend(['Training', 'Testing'], loc='upper right')
plt.show()
#plt.savefig('MNIST_loss_plot1.png')

plt.plot(model.history['acc'])
plt.plot(model.history['val_acc'])
plt.title('Accuracy Rate', size=14)
plt.ylabel('Accuracy %')
plt.xlabel('Training iterations')
plt.legend(['Training','Testing'], loc='lower right')
plt.show()
#plt.savefig('MNIST_acc_plot1.png')
'''

# Best Model is model2
print("\nTable II: Best parameters configuration for cross entropy loss function and relu activation function\n")
print(tabulate(results_df.iloc[[1]], headers='keys'))

# Train test confusion matrix
y_pred_train=classifier2.predict(X_train)
matrix_train = confusion_matrix(y_train.argmax(axis=1), y_pred_train.argmax(axis=1))
print("\nConfusion matrix for train set\n", matrix_train)

# Train Class accuracies
print("\nClass Acuracies (training):")
class_accuracy = np.zeros(10)
for x in range(0,10):
	true_positives = int(matrix_train[x,x])
	sum_per_class = 0
	for y in range(0,10):
		sum_per_class+=int(matrix_train[x,y])
	class_accuracy[x] = true_positives/sum_per_class
	print("Accuracy of class", x, "is = ", class_accuracy[x])

# Test set confusion matrix
y_pred_test=classifier2.predict(X_test)
matrix_test = confusion_matrix(y_test.argmax(axis=1), y_pred_test.argmax(axis=1))
print("\nConfusion matrix for test set:\n", matrix_test)


# Test Class accuracies
print("\nClass Acuracies (testing):")
class_accuracy = np.zeros(10)
for x in range(0,10):
	true_positives = int(matrix_test[x,x])
	sum_per_class = 0
	for y in range(0,10):
		sum_per_class+=int(matrix_test[x,y])
	class_accuracy[x] = true_positives/sum_per_class
	print("Accuracy of class", x, "is = ", class_accuracy[x])
