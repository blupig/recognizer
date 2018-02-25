# Train
import datetime
from keras.callbacks import TensorBoard
import data_input
import model

# Config
epochs = 150
workers = 6
data_path = 'tiny-imagenet-200'

# Build data generators
# Sample data for generators first
x_samples = data_input.load_sample_data(data_path)
print(x_samples[0][0])

train_gen, val_gen = data_input.data_generators(data_path, x_samples=x_samples)
for x_batch, y_batch in train_gen:
    print(x_batch[0][0])
    break

# Build and compile model
train_model, tmpl_model = model.cnn_model(gpus=2)

# Tensorboard
datetime_str = '{0:%y%m%d_%H%M}'.format(datetime.datetime.now())
tensorboard = TensorBoard(log_dir='logs/' + datetime_str)

# Train
train_model.fit_generator(
    train_gen,
    validation_data=val_gen,
    epochs=epochs,
    workers=workers,
    verbose=2,
    callbacks=[tensorboard])

# Save model
tmpl_model.save('model.h5')
