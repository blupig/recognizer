# CNN model
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import optimizers
from keras import metrics
from keras.utils import multi_gpu_model


def cnn_model(gpus=1):
    # Create new template model
    tmpl_model = Sequential()

    # In: [-1, 64, 64, 3]
    tmpl_model.add(Conv2D(filters=96, kernel_size=[5, 5], padding='same', input_shape=(64, 64, 3)))
    # tmpl_model.add(BatchNormalization())
    tmpl_model.add(Activation('relu'))

    tmpl_model.add(MaxPooling2D(pool_size=[2, 2]))

    # In: [-1, 32, 32, 96]
    tmpl_model.add(Conv2D(128, [3, 3], padding='same'))
    # tmpl_model.add(BatchNormalization())
    tmpl_model.add(Activation('relu'))

    tmpl_model.add(Conv2D(192, [3, 3], padding='same'))
    # tmpl_model.add(BatchNormalization())
    tmpl_model.add(Activation('relu'))

    tmpl_model.add(MaxPooling2D([2, 2]))

    # In: [-1, 16, 16, 192]
    tmpl_model.add(Conv2D(256, [3, 3], padding='same'))
    # tmpl_model.add(BatchNormalization())
    tmpl_model.add(Activation('relu'))

    tmpl_model.add(Conv2D(256, [3, 3], padding='same'))
    # tmpl_model.add(BatchNormalization())
    tmpl_model.add(Activation('relu'))

    tmpl_model.add(Conv2D(128, [3, 3], padding='same', activation='relu'))
    tmpl_model.add(MaxPooling2D([2, 2]))

    # Flatten to 1-D vector
    # In: [-1, 8, 8, 192]
    tmpl_model.add(Flatten())

    # Dense
    tmpl_model.add(Dense(units=1024, activation='relu'))

    # Dropout
    tmpl_model.add(Dropout(0.5))

    # Dense
    tmpl_model.add(Dense(units=512, activation='relu'))

    # Logits layer
    tmpl_model.add(Dense(units=20, activation='softmax'))

    # Optimizers
    adam = optimizers.Adam(lr=0.0001)
    sgd = optimizers.SGD(lr=0.01, decay=5e-6, momentum=0.9, nesterov=True)

    # The tmpl_model to be trained
    train_model = tmpl_model

    if gpus > 1:
        # Train on parallel tmpl_model
        train_model = multi_gpu_model(tmpl_model, gpus=gpus)

    # Compile modle
    train_model.compile(optimizer=sgd,  # sgd
                        loss='categorical_crossentropy',
                        metrics=['accuracy', metrics.top_k_categorical_accuracy])

    return train_model, tmpl_model
