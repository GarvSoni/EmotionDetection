from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D

def create_model():
    # Create the model

    # Creating a sequential model object
    model = Sequential()

    # Adding convolutional layers with 32 filters of size 3x3, using ReLU activation function and input shape of (48,48,1)
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))

    # Adding another convolutional layer with 64 filters of size 3x3, using ReLU activation function
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))

    # Adding a max pooling layer with pool size of 2x2 and a dropout layer with 25% dropout rate
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Adding convolutional layers with 128 filters of size 3x3, using ReLU activation function
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))

    # Adding a max pooling layer with pool size of 2x2 and a dropout layer with 25% dropout rate
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Adding convolutional layers with 256 filters of size 3x3, using ReLU activation function
    model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))

    # Adding a max pooling layer with pool size of 2x2 and a dropout layer with 25% dropout rate
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Adding a flatten layer to flatten the output from the convolutional layers
    model.add(Flatten())

    # Adding a dense layer with 512 neurons and using ReLU activation function, followed by a dropout layer with 50% dropout rate
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))

    # Adding a dense layer with 7 neurons (since there are 7 emotions to classify), using softmax activation function
    model.add(Dense(7, activation='softmax'))
    return model