from keras.preprocessing.image import ImageDataGenerator


# Create data generators with augmentation from directories
def data_generators():
    base_path = 'tiny-imagenet-200'

    train_datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    val_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_directory(
        directory='tiny-imagenet-200/train',
        target_size=(64, 64),
        batch_size=100,
        class_mode='sparse')

    val_gen = val_datagen.flow_from_directory(
        directory='tiny-imagenet-200/val',
        target_size=(64, 64),
        batch_size=100,
        class_mode='sparse')

    return train_gen, val_gen
