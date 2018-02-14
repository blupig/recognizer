# Imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import scipy
import tensorflow as tf
import model


def main(args):
    # Load training and eval data
    if len(args) < 2:
        print('ERROR: No data path provided')
        return

    print('Reading data...', end='', flush=True)

    # Init data arrays
    train_data = np.empty((0, 64, 64, 3), dtype=np.float32)
    train_labels = np.empty((0), dtype=np.int32)
    eval_data = np.empty((0, 64, 64, 3), dtype=np.float32)
    eval_labels = np.empty((0), dtype=np.int32)

    # Get a list of subdirectories in base_path, each is one class
    base_path = args[1]
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
        img_path_full = os.path.join(train_data_path, class_path, 'images')
        files = os.listdir(img_path_full)
        files = [os.path.join(img_path_full, f) for f in files]

        for i in range(0, 200):
            img = scipy.ndimage.imread(files[i], mode='RGB')
            img = img.reshape(1, 64, 64, 3)

            if i < 175:
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

    # Create the Estimator
    imagenet_classifier = tf.estimator.Estimator(model_fn=model.cnn_model_fn,
                                                 model_dir="tiny_imagenet_model")

    # Set up logging for predictions
    # tensors_to_log = {"probabilities": "softmax_tensor"}
    # logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log)

    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=100,
        num_epochs=None,
        shuffle=True)
    imagenet_classifier.train(input_fn=train_input_fn, steps=10000)

    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)
    eval_results = imagenet_classifier.evaluate(input_fn=eval_input_fn)

    print(eval_results)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
