# Train
from time import time
from keras.callbacks import TensorBoard
import data_input
import model

# Config
epochs = 150
workers = 6
data_path = 'tiny-imagenet-200'

# Build data generators
# Sample data for generators first
# x_samples = data_input.load_sample_data(data_path)
# print(x_samples[0])
train_gen, val_gen = data_input.data_generators(data_path)
# for x_batch, y_batch in train_gen:
#     print(x_batch[0])
#     break


# Build and compile model
m = model.cnn_model()

# Tensorboard
tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

# Train
m.fit_generator(
    train_gen,
    validation_data=val_gen,
    epochs=epochs,
    workers=workers,
    callbacks=[tensorboard])

# Save model
m.save('model.h5')
