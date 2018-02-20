# Data input
from os import path
from keras.preprocessing.image import ImageDataGenerator


# Config
# Read sample images
def load_sample_data(base_path):
    print('[PROG] Sampling data for fitting ImageDataGenerator...')
    sample_datagen = ImageDataGenerator()
    sample_gen = sample_datagen.flow_from_directory(
        directory=path.join(base_path, 'train'),
        target_size=(64, 64),
        batch_size=5000,
        class_mode='categorical')

    for x_batch, y_batch in sample_gen:
        print('[PROG] Done sampling data')
        return x_batch

    return None


# Create data generators with augmentation from directories
def data_generators(base_path, x_samples=None):

    train_datagen = ImageDataGenerator(
        # featurewise_center=True,
        # featurewise_std_normalization=True,
        # zca_whitening=True,
        rotation_range=60,
        width_shift_range=0.3,
        height_shift_range=0.3,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    if x_samples is not None:
        train_datagen.fit(x_samples)

    val_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_directory(
        directory=path.join(base_path, 'train'),
        target_size=(64, 64),
        batch_size=100,
        class_mode='categorical')

    val_gen = val_datagen.flow_from_directory(
        directory=path.join(base_path, 'val'),
        target_size=(64, 64),
        batch_size=100,
        class_mode='categorical')

    return train_gen, val_gen
