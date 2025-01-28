import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay

print(f"TensorFlow Version: {tf.__version__}")

# Load the training and testing datasets
train_csv = "fer2013_train.csv"
test_csv = "fer2013_test.csv"

# Load the data
train_data = pd.read_csv(train_csv)
test_data = pd.read_csv(test_csv)

# Separate features and labels
X_train = train_data.iloc[:, :-7].values
Y_train = train_data.iloc[:, -7:].values

X_test = test_data.iloc[:, :-7].values
Y_test = test_data.iloc[:, -7:].values

# Reshape features into 48x48 grayscale images
X_train = X_train.reshape(-1, 48, 48, 1).astype('float32') / 255.0
X_test = X_test.reshape(-1, 48, 48, 1).astype('float32') / 255.0

# Split training data into training and validation sets
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.1, random_state=42)

# Data augmentation
datagen = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=10,
        zoom_range = 0.0,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=False,
        vertical_flip=False)
datagen.fit(X_train)

# Build the CNN model
model = Sequential()

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.22))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))


# Define exponential decay schedule
initial_lr = 0.001
lr_schedule = ExponentialDecay(
    initial_learning_rate=initial_lr,
    decay_steps=1000,
    decay_rate=0.96,
    staircase=True
)

# Create Adam optimizer
custom_adam = Adam(learning_rate=lr_schedule)

# Compile the model
model.compile(optimizer=custom_adam, loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    datagen.flow(X_train, Y_train, batch_size=64),
    validation_data=(X_val, Y_val),
    epochs=50,
    shuffle=True,
    steps_per_epoch=len(X_train) // 64,
    verbose=1
)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(X_test, Y_test, verbose=1)
print(f"Test Accuracy: {test_acc * 100:.2f}%")

# Save the model
model.save("facial_expression_model2.keras")
print("Model saved as facial_expression_model2.keras")