# Imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import scipy.ndimage
import sys

# Load training and eval data
if len(sys.argv) < 2:
    print('ERROR: No data path provided')
    sys.exit(1)

print('Reading data...', end='', flush=True)

# Init data arrays
train_data = np.empty((0, 64, 64, 3), dtype=np.float32)
train_labels = np.empty((0), dtype=np.int32)
eval_data = np.empty((0, 64, 64, 3), dtype=np.float32)
eval_labels = np.empty((0), dtype=np.int32)

# Get a list of subdirectories in base_path, each is one class
base_path = sys.argv[1]
train_data_path = os.path.join(base_path, 'train')
class_paths = []
for _, dirs, _ in os.walk(train_data_path):
    class_paths = dirs
    break  # Only need first level

# Sort paths
class_paths.sort()

# Read images and assign labels
class_id = 0
for class_path in class_paths:
    # Images base path
    img_path_full = os.path.join(train_data_path, class_path, 'images')

    # All images
    files = os.listdir(img_path_full)
    files = [os.path.join(img_path_full, f) for f in files]

    for i in range(0, 450):
        # Decode image
        img = scipy.ndimage.imread(files[i], mode='RGB')
        img = img.reshape(1, 64, 64, 3)

        if i < 400:
            # Read first some files as training data
            train_data = np.append(train_data, img, axis=0)
            train_labels = np.append(train_labels, class_id)
        else:
            # Read some more for evaluation data
            eval_data = np.append(eval_data, img, axis=0)
            eval_labels = np.append(eval_labels, class_id)

    class_id += 1
    if class_id == 10:
        break

print('done', flush=True)
print('train_data: ', train_data.shape)
print('eval_data: ', eval_data.shape)

print('Saving arrays...', end='', flush=True)
np.save(os.path.join(base_path, 'train_data'), train_data)
np.save(os.path.join(base_path, 'train_labels'), train_labels)
np.save(os.path.join(base_path, 'eval_data'), eval_data)
np.save(os.path.join(base_path, 'eval_labels'), eval_labels)
print('done', flush=True)
