from tensorflow.keras.preprocessing.image import ImageDataGenerator

def dataset(train_dir, val_dir, batch_size):
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
    return train_generator, validation_generator