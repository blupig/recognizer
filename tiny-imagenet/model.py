from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense


def cnn_model():
    model = Sequential()

    # Conv + pooling #1
    model.add(Conv2D(filters=32, kernel_size=[5, 5], padding='same', activation='relu', input_shape=(64, 64, 3)))
    model.add(MaxPooling2D(pool_size=[2, 2]))

    # Conv + pooling #2
    model.add(Conv2D(64, [3, 3], padding='same', activation='relu'))
    model.add(MaxPooling2D([2, 2]))

    # Conv + pooling #3
    model.add(Conv2D(128, [3, 3], padding='same', activation='relu'))
    model.add(Conv2D(128, [3, 3], padding='same', activation='relu'))
    model.add(MaxPooling2D([2, 2]))

    # Flatten to 1-D vector
    model.add(Flatten())

    # Dense layer #1
    model.add(Dense(units=512, activation='relu'))

    # Dropout
    model.add(Dropout(0.5))

    # Logits layer
    model.add(Dense(units=20))

    model.compile(optimizer='adam',  # sgd
                  loss='sparse_categorical_crossentropy',  # categorical_crossentropy
                  metrics=['accuracy'])

    return model
