# Data input
from os import path
import numpy as np
from keras.preprocessing.image import ImageDataGenerator


# Read sample images
def load_sample_data(base_path):
    print('[PROG] Sampling data for fitting ImageDataGenerator...')
    sample_datagen = ImageDataGenerator()
    sample_gen = sample_datagen.flow_from_directory(
        directory=path.join(base_path, 'train'),
        target_size=(64, 64),
        batch_size=1000,
        class_mode='categorical')

    for x_batch, y_batch in sample_gen:
        print('[PROG] Done sampling data')
        return x_batch

    return None


# Create data generators with augmentation from directories
def data_generators(base_path, x_samples):

    train_datagen = ImageDataGenerator(
        featurewise_center=False,
        # zca_whitening=True,
        rotation_range=45,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    val_datagen = ImageDataGenerator(
        featurewise_center=False,
        rescale=1./255)

    # Fit generator if x samples provided
    if x_samples is not None:
        train_datagen.fit(x_samples / 255)
        val_datagen.fit(x_samples / 255)

    train_gen = train_datagen.flow_from_directory(
        directory=path.join(base_path, 'train'),
        target_size=(64, 64),
        batch_size=512,
        class_mode='categorical')

    val_gen = val_datagen.flow_from_directory(
        directory=path.join(base_path, 'val'),
        target_size=(64, 64),
        batch_size=512,
        class_mode='categorical')

    return train_gen, val_gen


def predict_generator(base_path, x_samples):
    predict_datagen = ImageDataGenerator(
        featurewise_center=False,
        rescale=1./255)

    # Fit generator if x samples provided
    if x_samples is not None:
        predict_datagen.fit(x_samples / 255)

    predict_gen = predict_datagen.flow_from_directory(
        directory=path.join(base_path, 'predict'),
        target_size=(64, 64),
        batch_size=512,
        class_mode=None,
        shuffle=False)

    return predict_gen


# Read files from a generator, return array and file names
def read_files_into_memory(gen):
    filenames = gen.filenames
    for i in range(len(gen)):
        x_batch = next(gen)
        # Highly inefficient append operation
        if 'predit_x' not in locals():
            predit_x = x_batch
        else:
            predit_x = np.append(predit_x, x_batch, axis=0)

    return predit_x, filenames
