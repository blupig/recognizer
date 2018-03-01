# Train
import sys
import datetime
from keras.callbacks import TensorBoard
import data_input
import model

# Config
epochs = 200
workers = 8
data_path = 'tiny-imagenet-200'

# Sample data for generators
# x_samples = data_input.load_sample_data(data_path)
# for n in x_samples[0][0]:
#     print(n, end=' ')

# print()  # New line

# Build data generators
train_gen, val_gen = data_input.data_generators(data_path, x_samples=None)
for x_batch, y_batch in train_gen:
    for n in x_batch[0][0]:
        print(n, end=' ')
    print()  # New line
    break

# Build and compile model
train_model, tmpl_model = model.cnn_model(gpus=2)

# Tensorboard
run_comment = '0'
if len(sys.argv) > 1:
    run_comment = sys.argv[1]

datetime_str = '{0:%y%m%d_%H%M}'.format(datetime.datetime.now())
tensorboard = TensorBoard(log_dir='logs/' + datetime_str + '_' + run_comment)

# Train
train_model.fit_generator(
    train_gen,
    validation_data=val_gen,
    epochs=epochs,
    workers=workers,
    use_multiprocessing=True,
    verbose=2,
    callbacks=[tensorboard])

# Save model
tmpl_model.save_weights('model.h5')
