import numpy as np
import argparse
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
import os
import config
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# command line argument
ap = argparse.ArgumentParser()
ap.add_argument("--whattodo",help="train_it/test_it")
whattodo = ap.parse_args().whattodo

# plots accuracy and loss curves
def plot_model_history(model_history):
    """
    Plot Accuracy and Loss curves given the model_history
    """
    fig, axs = plt.subplots(1,2,figsize=(15,5))
    # summarize history for accuracy
    axs[0].plot(range(1,len(model_history.history['accuracy'])+1),model_history.history['accuracy'])
    axs[0].plot(range(1,len(model_history.history['val_accuracy'])+1),model_history.history['val_accuracy'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1,len(model_history.history['accuracy'])+1),len(model_history.history['accuracy'])/10)
    axs[0].legend(['train', 'val'], loc='best')
    # summarize history for loss
    axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])
    axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1,len(model_history.history['loss'])+1),len(model_history.history['loss'])/10)
    axs[1].legend(['train', 'val'], loc='best')
    fig.savefig('plot.png')
    plt.show()

# Define data generators
train_dir = config.train_dir
val_dir = config.val_dir
# val_dir = 'data/test'

num_train = 28709
num_val = 7178
batch_size = 128
num_epoch = 45 # we got great accuracy till 45 (89%) after this model starts overfitting.
# This observation is totally depend on the dataset you use.

train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(48,48),
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode='categorical')

validation_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(48,48),
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode='categorical')

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

# If you want to train the same model or try other models, go for this
# Check if mode is "train"
if whattodo == "train_it":
    
    # Check if a pre-existing model weights file exists and load it if it does
    if os.path.isfile(config.weights_dir):
        print('Loading model weights...')
        model.load_weights(config.weights_dir)
    
        # Compile the model using categorical crossentropy loss function, Adam optimizer with learning rate of 0.0001 and decay rate of 1e-6, and accuracy metric
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001, decay=1e-6), metrics=['accuracy'])
    
    # Set up a ModelCheckpoint callback to save the best model weights during training
    checkpoint_callback = ModelCheckpoint(config.weights_dir, monitor='val_accuracy', save_best_only=True)
    
    # Set up a TensorBoard callback to log training progress
    tensorboard_callback = TensorBoard(log_dir=config.logs_dir, update_freq='epoch', write_graph=False)
    
    # Train the model using a generator for the training data and a generator for the validation data, with a specified number of steps per epoch, epochs, and validation steps, and with the ModelCheckpoint and TensorBoard callbacks
    model_info = model.fit_generator(
        train_generator,
        steps_per_epoch=num_train // batch_size,
        epochs=num_epoch,
        validation_data=validation_generator,
        validation_steps=num_val // batch_size,
        callbacks=[tensorboard_callback, checkpoint_callback])
    plot_model_history(model_info)


# emotions will be displayed on your face from the webcam feed# Check if mode is "display"
elif whattodo == "test_it":
    
    # Load the pre-trained model weights
    model.load_weights(config.weights_dir)

    # Turn off OpenCL usage and logging messages
    cv2.ocl.setUseOpenCL(False)

    # Create a dictionary which maps each label to an emotion in alphabetical order
    emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

    # Start the webcam feed
    cap = cv2.VideoCapture(0)
    while True:
        # Detect faces using a Haar cascade classifier
        ret, frame = cap.read()
        if not ret:
            break
        facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facecasc.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        # For each detected face, draw a bounding box and predict the emotion
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 225, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
            prediction = model.predict(cropped_img)
            maxindex = int(np.argmax(prediction))
            cv2.putText(frame, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Display the webcam feed with bounding boxes and predicted emotions
        cv2.imshow('Video', cv2.resize(frame, (1600, 960), interpolation=cv2.INTER_CUBIC))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()
