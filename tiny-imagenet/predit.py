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

# Build data generators
predict_gen = data_input.predict_generator(base_path, x_samples=None)

# Read data
predit_x, filenames = data_input.read_files_into_memory(predict_gen)
print(predit_x[0][0])

# Build and compile model
print('Loading model...')
predict_model, _ = model.cnn_model(gpus=0)
predict_model.load_weights('model_weights_180301.h5')

# Train
print('Predicting...')
results = predict_model.predict(predit_x, batch_size=1, verbose=1)

# Get class names
cls_names = class_names.get_names(base_path)

# Walk through results
for i in range(len(results)):
    filename = filenames[i]
    probabilities = results[i]
    print('--- ' + filename)

    # Sort array in descending order, retrieve indexes (class_ids)
    sorted_idx = np.argsort(probabilities)[::-1]

    k = 0
    for clsid in sorted_idx:
        print(clsid, cls_names[clsid], probabilities[clsid])
        # Only top 5
        k += 1
        if k >= 5:
            break
    print('')
