import numpy as np
import pandas as pd
from sklearn.svm import SVC

# Loading file data
print('Loading data files as given.....\n\n')
x_train= np.load('X_train.npy')
y_train = np.load('Y_train.npy')
x_test = np.load('X_test.npy')

#Understanding the dimenison of data
print('Finding out the dimension of given data.....\n')
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)

#Reshaping of the data. No reshaping is required for y_train as its single dimension
print('\n\nReshaping Data..... \n\n')
x_train_data = x_train.reshape(4160,50*50*3)
x_test_data = x_test.reshape(1387,50*50*3)

#Building Classifier using SVM
print('Building Classifier..... \n\n')
classifier = SVC(kernel='rbf', random_state = 1)
classifier.fit(x_train_data,y_train)

#Prediction for x_test using Classifier
print('Predicting Values for X_test..... \n\n')      
Y_predictedValue = classifier.predict(x_test_data)
print(Y_predictedValue)

#Saving output as CSV
np.savetxt('anchal_gupta.csv', Y_predictedValue,fmt='%.2f')




