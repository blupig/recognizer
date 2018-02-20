from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import optimizers
from keras import metrics


def cnn_model():
    model = Sequential()

    # Out: [-1, 32, 32, 96]
    model.add(Conv2D(filters=96, kernel_size=[5, 5], padding='same', activation='relu', input_shape=(64, 64, 3)))
    model.add(MaxPooling2D(pool_size=[2, 2]))

    # Out: [-1, 16, 16, 192]
    model.add(Conv2D(128, [3, 3], padding='same', activation='relu'))
    model.add(Conv2D(192, [3, 3], padding='same', activation='relu'))
    model.add(MaxPooling2D([2, 2]))

    # Out: [-1, 16, 16, 256]
    model.add(Conv2D(256, [3, 3], padding='same', activation='relu'))
    # model.add(MaxPooling2D([2, 2]))

    # Out: [-1, 8, 8, 256]
    model.add(Conv2D(256, [3, 3], padding='same', activation='relu'))
    model.add(Conv2D(256, [3, 3], padding='same', activation='relu'))
    model.add(MaxPooling2D([2, 2]))

    # Flatten to 1-D vector
    model.add(Flatten())

    model.add(Dense(units=1024, activation='relu'))

    # Dropout
    model.add(Dropout(0.5))

    # Logits layer
    model.add(Dense(units=20, activation='softmax'))

    # Optimizers
    adam = optimizers.Adam(lr=0.0001)
    sgd = optimizers.SGD(lr=0.01, decay=5e-6, momentum=0.9, nesterov=True)

    # Compile modle
    model.compile(optimizer=sgd,  # sgd
                  loss='categorical_crossentropy',
                  metrics=['accuracy', metrics.top_k_categorical_accuracy])

    return model
