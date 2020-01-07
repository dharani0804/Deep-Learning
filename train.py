from __future__ import print_function
import tensorflow as tf
import keras
import argparse
from keras import datasets
from keras import layers
from keras import models
from keras import utils
from keras.layers import Dense
from keras.models import Sequential
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Activation
from keras.regularizers import l2
from keras.optimizers import SGD
from keras.optimizers import RMSprop
from keras.callbacks import History
from keras import losses
from keras import backend as K
import mlflow
import numpy as np

mlflow.set_tracking_uri('https://104.197.227.88:5000')

mlflow.create_experiment('your_experiment_name')

mlflow.set_experiment('your_experiment_name')
mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0

# reshape the data into 1D vectors
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)

num_classes = 10

# Check the column length
x_train.shape[1]

num_classes = 10
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


np.random.seed(40)

parser = argparse.ArgumentParser()
parser.add_argument('--epochs')
parser.add_argument('--learning_rate')
parser.add_argument('--batch_size')
args = parser.parse_args()


P1= int(args.epochs) ## P1 = EPOCHS
P2 = float(args.learning_rate)  ## P2 = LEARNING RATE
input_dim = x_train.shape[1]
P3 = int(args.batch_size) #BATCH SIZE
P4 = P2/ P1 ## P4 = DECAY


momentum = 0.8

model = Sequential()
model.add(Dense(64, activation='relu', kernel_initializer='uniform', 
                input_dim = input_dim)) 
model.add(Dropout(0.1))
model.add(Dense(64, kernel_initializer='uniform', activation='relu'))
model.add(Dense(num_classes, kernel_initializer='uniform', activation='sigmoid'))
sgd = SGD(lr=P2, decay=P4,nesterov=False)

y_pred = model.predict(x_test)
y_pred =(y_pred>0.5)
list(y_pred)


def recall_model(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

def precision_model(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

def f1_model(y_true, y_pred):
    precision = precision_model(y_true, y_pred)
    recall = recall_model(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


with mlflow.start_run():
    # compile the model
    model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['acc',f1_model,precision_model, recall_model])

    # fit the model
    history = model.fit(x_train, y_train, validation_split=0.3, epochs= P1, batch_size = P3, verbose=1)

    # evaluate the model
    loss, accuracy, f1_model, precision_model, recall_model = model.evaluate(x_test, y_test, verbose=1)
    mlflow.log_param('P1',P1)
    mlflow.log_param('P2',P2)
    mlflow.log_param('P3',P3)
    mlflow.log_metric('loss',loss)
    mlflow.log_metric('accuracy',accuracy)
    mlflow.log_metric('f1_model',f1_model)
    mlflow.log_metric('precision_model',precision_model)
    mlflow.log_metric('recall_model',recall_model)
    mlflow.end_run()


             

