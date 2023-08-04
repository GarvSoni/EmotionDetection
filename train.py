import config
import os 
from tensorflow.keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import EarlyStopping


def train(model, train_generator, num_train, batch_size, num_epoch,validation_generator, num_val, plot_model_history):
    # Check if a pre-existing model weights file exists and load it if it does
    # if os.path.isfile(config.weights_dir):
    #     print('Loading model weights...')
    #     model.load_weights(config.weights_dir)

    # Compile the model using categorical crossentropy loss function, Adam optimizer with learning rate of 0.0001 and decay rate of 1e-6, and accuracy metric
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001, decay=1e-6), metrics=['accuracy'])

    # Set up a ModelCheckpoint callback to save the best model weights during training
    checkpoint_callback = ModelCheckpoint(config.weights_dir, monitor='val_accuracy', save_best_only=True)


    # Set up a ModelCheckpoint callback to save model weights after each epoch
    checkpoint_callback = ModelCheckpoint(config.weights_dir, monitor='val_accuracy', save_best_only=False)

    # Set up EarlyStopping to prevent overfitting
    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Train the model using a generator for the training data and a generator for the validation data, with a specified number of steps per epoch, epochs, and validation steps, and with the ModelCheckpoint, TensorBoard, and EarlyStopping callbacks
    model_info = model.fit_generator(
        train_generator,
        steps_per_epoch=num_train // batch_size,
        epochs=num_epoch,
        validation_data=validation_generator,
        validation_steps=num_val // batch_size,
        callbacks=[checkpoint_callback, checkpoint_callback, early_stopping_callback])
    plot_model_history(model_info)