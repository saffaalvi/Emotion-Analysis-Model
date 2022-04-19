import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from keras.utils import np_utils
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dropout, Flatten, Dense
import joblib 

# Get the dataset
data_features = pd.read_csv('data_features.csv')

# Features to train off of
features = data_features.iloc[:,2:]

# Target values (emotions)
targets = data_features.iloc[:,1:2]

# Split into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=1)
print("The size of the training set is:", x_train.shape)
print("The size of the testing set is:", x_test.shape)

# Label Encoding the targets (Emotions -> Numbers)
lb = LabelEncoder()
y_train = np_utils.to_categorical(lb.fit_transform(y_train.values.ravel()))
y_test = np_utils.to_categorical(lb.fit_transform(y_test.values.ravel()))

# Save the labels to a file so we can use them later for predictions
labels = lb.classes_
np.save('labels.npy', lb.classes_)

# Scale/Normalize data features 
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train.to_numpy())
x_test = scaler.transform(x_test)

# Save the scalar to use with future predictions
joblib.dump(scaler, 'scaler.save')
print("Saved scaler!")

# Change dimensions
x_train = np.expand_dims(x_train, axis=2)
x_test = np.expand_dims(x_test, axis=2)

# ----------------------- CONVOLUTIONAL NEURAL NETWORK MODEL -----------------------
model = Sequential()
model.add(Conv1D(32, kernel_size=(5), input_shape=(40, 1), activation='relu'))
model.add(Conv1D(64, kernel_size=(5), activation='relu'))
model.add(Conv1D(128, kernel_size=(5), activation='relu'))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation ='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation ='relu'))
model.add(Dense(8, activation='softmax'))

# ----------------------- CONVOLUTIONAL NEURAL NETWORK MODEL -----------------------

# Compile the model 
opt = tf.keras.optimizers.Adam()
model.compile(optimizer=opt, loss ='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train the model
fit = model.fit(x_train, y_train, epochs = 25, validation_data = (x_test, y_test))

# Save the model
model.save('model.h5')
json_model = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(json_model)

print("Saved model!")
