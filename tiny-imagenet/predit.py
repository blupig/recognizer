# Train
import sys
import numpy as np
from keras.models import load_model
import data_input
import model
import class_names

# Config
# Data path
base_path = 'tiny-imagenet-200'

# Sample data for generators
# x_samples = data_input.load_sample_data(base_path)
# for n in x_samples[0][0]:
#     print(n, end=' ')

# print()  # New line

# Build data generators
predict_gen = data_input.predict_generator(base_path, x_samples=None)
for x_batch in predict_gen:
    for n in x_batch[0][0]:
        print(n, end=' ')
    print()  # New line
    break

# Build and compile model
print('Loading model...')
predict_model, _ = model.cnn_model(gpus=0)
predict_model.load_weights('model.h5')

# Train
print('Predicting...')
results = predict_model.predict_generator(
    predict_gen,
    workers=1,
    use_multiprocessing=False,
    verbose=1)

c_names = class_names.get_names(base_path)

# Walk through results
for r in results:
    print('---')
    # Sort array in descending order, retrieve indexes
    sorted_idx = np.argsort(r)[::-1]

    k = 0
    for i in sorted_idx:
        print(i, c_names[i], r[i])
        # Only top 5
        k += 1
        if k >= 5:
            break
    print('')
