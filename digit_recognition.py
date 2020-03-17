import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

## loading of data 
train_data = np.loadtxt("Datasets/DigitTrain.csv", delimiter=',',skiprows=1)
test_data = np.loadtxt("Datasets/DigitTest.csv", delimiter=',',skiprows=1)

## visualising how data looks
plt.imshow(test_data[10].reshape((28,28)))
plt.show()

## forming x_train and y_train
x_train = train_data[:,1:]
y_train = train_data[:,0]

print(x_train.shape, y_train.shape) ## to check the size of data

## feature scaling of daat
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
test_data = scaler.transform(test_data)



from keras.utils.np_utils import to_categorical

## converting y_train into one hot encoded form to be suitable for the model
y_train = to_categorical(y_train, num_classes= 10)

## reshaping x_train and x_test
x_train = np.reshape(x_train, (42000,28,28,1))
x_test = np.reshape(test_data, (28000,28,28,1))

print(x_train.shape, y_train.shape) ## to check the size of data



from sklearn.model_selection import train_test_split
X_train, X_val, Y_train, Y_val = train_test_split(x_train, y_train, test_size = 0.3)


## Building the model using keras

from keras.layers import Conv2D, Dense, MaxPooling2D, Flatten, Dropout
from keras.models import Sequential

model = Sequential() ## creating a model
 
## first conv 2d layer 
Conv1 = Conv2D(filters = 32, kernel_size = (3,3), padding="same",input_shape = (28,28,1), activation = "relu")
model.add(Conv1)

## max_pooling layer
max_pool1 = MaxPooling2D(pool_size=(2, 2), padding='same')
model.add(max_pool1)


## second conv 2d layer
Conv2 = Conv2D(filters = 32, kernel_size = (3,3),strides = (1,1) , padding = "same", activation = "relu")
model.add(Conv2)

## max_pooling layer
max_pool2 = MaxPooling2D(pool_size=(2, 2), padding='same')
model.add(max_pool2)

## flattening the final output from cnn layers before feeding it to the dense layers
model.add(Flatten())


## building the neural network model for classification

layer1 = Dense(units = 10, activation = "relu")
model.add(layer1)

## 10 digits , hence output layer contains 10 units
output_layer = Dense(units = 10, activation = "sigmoid")
model.add(output_layer)

## drop out layer added to reduce overfitting and improve accuracy
model.add(Dropout(0.10))

## compilling the model
model.compile(optimizer = "adam",loss = "binary_crossentropy", metrics = ["accuracy"])

## fitting  the data
model.fit(x_train, y_train, epochs = 5, batch_size = 50, validation_data = (X_val, Y_val) )

## printing the model designed
model.summary()

## final prediction using the test data 
y_pred = model.predict(x_test)

## converting data into suitable form for submission on Kaggle
Y_pred_classes = np.argmax(y_pred, axis = 1) 
Y_pred_classes
df = pd.DataFrame(np.arange(1,len(Y_pred_classes)+1), columns = ["ImageID"])
df["Label"] = Y_pred_classes

## saving the predictions into csv format
df.to_csv("preds.csv", index = False)